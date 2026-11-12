"""
DriftBlimp UI server
Sits between the iPad app and the home base receiver (homebase_reciever.py).
It also stores the item list and the item reference photos.

MOCK mode (default): pretends to be the home base. No blimp needed.
REAL mode: forwards searches to the real receiver on port 5000.
           Needs the receiver to have GET /classes and POST /target
           (and optionally GET /found_image).

Run from the ui folder:
    pip install fastapi uvicorn requests
    python -m uvicorn ui_server:app --host 0.0.0.0 --port 8000

Real mode (PowerShell):
    $env:DRIFT_MODE="real"
    python -m uvicorn ui_server:app --host 0.0.0.0 --port 8000

Item data lives in ui/data/ (items.json + photos/). That folder is
git-ignored automatically so personal photos never reach GitHub.
"""
import json
import os
import threading
import time

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

MODE = os.environ.get("DRIFT_MODE", "mock").lower()          # "mock" or "real"
HB = os.environ.get("HOMEBASE_URL", "http://localhost:5000")  # the real receiver
HERE = os.path.dirname(os.path.abspath(__file__))
STEP_SECONDS = 2.5
MOCK_CLASSES = ["keys", "glasses", "phone", "tv remote", "pill box"]
CATEGORIES = ["Everyday", "Medicine", "Other"]

app = FastAPI()

# ---------------------------------------------------------------- item storage
DATA = os.path.join(HERE, "data")
PHOTOS = os.path.join(DATA, "photos")
ITEMS_FILE = os.path.join(DATA, "items.json")
os.makedirs(PHOTOS, exist_ok=True)
_gi = os.path.join(DATA, ".gitignore")
if not os.path.exists(_gi):                  # keep personal photos out of GitHub
    with open(_gi, "w") as f:
        f.write("*\n")

LOCK = threading.Lock()


def _seed():
    names = [("Keys", "Everyday", "keys"), ("Glasses", "Everyday", "glasses"),
             ("Phone", "Everyday", "phone"), ("TV remote", "Everyday", "tv remote"),
             ("Pill box", "Medicine", "pill box")]
    return [{"id": n + 1, "name": a, "cat": b, "model_class": c, "notes": "",
             "last_found": None} for n, (a, b, c) in enumerate(names)]


def _load():
    try:
        with open(ITEMS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return _seed()


def _save():
    tmp = ITEMS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ITEMS, f, indent=2)
    os.replace(tmp, ITEMS_FILE)              # safe write: never a half-written file


ITEMS = _load()
_save()


def _photo_path(item_id, slot):
    return os.path.join(PHOTOS, f"{int(item_id)}_{int(slot)}.jpg")


def _public(item):
    out = dict(item)
    urls = []
    for s in range(3):
        p = _photo_path(item["id"], s)
        urls.append(f"/api/items/{item['id']}/photo/{s}?v={int(os.path.getmtime(p))}"
                    if os.path.exists(p) else None)
    out["photos"] = urls
    return out


def _find(item_id):
    for it in ITEMS:
        if it["id"] == item_id:
            return it
    raise HTTPException(404, "Item not found.")


class ItemBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    cat: str = "Everyday"
    model_class: str = Field(default="", max_length=60)
    notes: str = Field(default="", max_length=500)


def _clean(body):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Please give the item a name.")
    return {"name": name, "cat": body.cat if body.cat in CATEGORIES else "Other",
            "model_class": body.model_class.strip(), "notes": body.notes.strip()}


@app.get("/api/items")
def list_items():
    with LOCK:
        return [_public(i) for i in ITEMS]


@app.post("/api/items")
def add_item(body: ItemBody):
    with LOCK:
        item = {"id": max([i["id"] for i in ITEMS] + [0]) + 1,
                "last_found": None, **_clean(body)}
        ITEMS.append(item)
        _save()
        return _public(item)


@app.put("/api/items/{item_id}")
def edit_item(item_id: int, body: ItemBody):
    with LOCK:
        item = _find(item_id)
        item.update(_clean(body))
        _save()
        return _public(item)


@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    with LOCK:
        item = _find(item_id)
        ITEMS.remove(item)
        _save()
        for s in range(3):
            try:
                os.remove(_photo_path(item_id, s))
            except OSError:
                pass
    return {"ok": True}


@app.put("/api/items/{item_id}/photo/{slot}")
async def put_photo(item_id: int, slot: int, request: Request):
    """Body = the raw JPEG bytes. Slots: 0 front, 1 top, 2 side."""
    with LOCK:
        _find(item_id)
    if slot not in (0, 1, 2):
        raise HTTPException(400, "Photo slot must be 0, 1 or 2.")
    data = await request.body()
    if len(data) > 5_000_000 or not data.startswith(b"\xff\xd8"):
        raise HTTPException(400, "Photo must be a JPEG under 5 MB.")
    with open(_photo_path(item_id, slot), "wb") as f:
        f.write(data)
    return {"ok": True}


@app.get("/api/items/{item_id}/photo/{slot}")
def get_photo(item_id: int, slot: int):
    p = _photo_path(item_id, slot)
    if slot not in (0, 1, 2) or not os.path.exists(p):
        raise HTTPException(404, "No photo.")
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})


# ------------------------------------------------------------ blimp / home base
# mode: IDLE, SEARCHING, RETURNING, DOCKED, ESTOPPED, MAPPING
STATE = {"mode": "IDLE", "item_id": None, "t0": 0.0, "before_stop": None,
         "map_pct": 0, "battery": 85, "found": False, "baseline": 0,
         "hb_ok": True}


class SearchRequest(BaseModel):
    item_id: int
    confidence: float = 0.5


def _elapsed():
    return time.time() - STATE["t0"]


def _set(mode):
    STATE["mode"] = mode
    STATE["t0"] = time.time()


def _mark_found():
    """Remember when the item was found (shown on the My Items screen)."""
    STATE["found"] = True
    with LOCK:
        for it in ITEMS:
            if it["id"] == STATE["item_id"]:
                it["last_found"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "room": None}   # room names come with mapping
                _save()


def _homebase(method, path, **kw):
    """Talk to the real receiver. Raises HTTPException if it can't."""
    try:
        r = requests.request(method, HB + path, timeout=3, **kw)
        STATE["hb_ok"] = True
        return r
    except requests.RequestException:
        STATE["hb_ok"] = False
        raise HTTPException(502, "Can't reach the home base.")


@app.get("/api/status")
def status():
    """The UI asks this about once a second."""
    mode = STATE["mode"]
    stage = 0

    if mode == "SEARCHING":
        if MODE == "real":
            stage = min(2, int(_elapsed() // STEP_SECONDS))
            try:
                data = _homebase("GET", "/status").json()
                if data["event_id"] > STATE["baseline"]:      # new detection!
                    _mark_found()
                    _set("RETURNING")
                    mode, stage = "RETURNING", 0
            except (HTTPException, ValueError, KeyError):
                pass
        else:                                                  # MOCK
            stage = int(_elapsed() // STEP_SECONDS)
            if stage >= 3:
                _mark_found()
                _set("RETURNING")
                mode, stage = "RETURNING", 0

    if mode == "RETURNING":          # SIMULATED in both modes (no flight code yet)
        stage = min(3, int(_elapsed() // STEP_SECONDS))
        if stage >= 3:
            mode = STATE["mode"] = "DOCKED"

    if mode == "MAPPING":            # SIMULATED in both modes
        STATE["map_pct"] = min(100, int(_elapsed() / 0.7) * 4)

    return {"mode": mode, "stage": stage, "item_id": STATE["item_id"],
            "found": STATE["found"], "battery": STATE["battery"],
            "map_pct": STATE["map_pct"], "demo": MODE != "real",
            "homebase_ok": STATE["hb_ok"]}


@app.get("/api/classes")
def classes():
    """Names the home base's AI model knows (for the Add Item screen)."""
    if MODE == "real":
        try:
            return _homebase("GET", "/classes").json()
        except (HTTPException, ValueError):
            return []
    return MOCK_CLASSES


@app.get("/api/found_image")
def found_image():
    """Cropped photo of the found object, if the receiver provides one."""
    if MODE != "real":
        raise HTTPException(404, "No image.")
    r = _homebase("GET", "/found_image")
    if r.status_code != 200:
        raise HTTPException(404, "No image.")
    return Response(r.content, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=300"})


@app.post("/api/search")
def search(req: SearchRequest):
    with LOCK:
        item = dict(_find(req.item_id))
    target = (item["model_class"] or item["name"]).strip()
    if MODE == "real":
        r = _homebase("POST", "/target",
                      json={"name": target, "confidence": req.confidence})
        if r.status_code == 400:
            raise HTTPException(
                400, f"The home base doesn't know how to find \"{item['name']}\" yet. "
                     "A caregiver can set the AI name in Edit.")
        if not r.ok:
            raise HTTPException(502, "The home base had a problem.")
        STATE["baseline"] = _homebase("GET", "/status").json()["event_id"]
    STATE["item_id"] = req.item_id
    STATE["found"] = False
    _set("SEARCHING")
    return {"ok": True, "message": "Command received"}


@app.post("/api/return_home")
def return_home():
    _set("RETURNING")       # REAL VERSION: send a return command toward the blimp
    return {"ok": True, "message": "Command received"}


@app.post("/api/estop")
def estop():
    if STATE["mode"] != "ESTOPPED":
        STATE["before_stop"] = STATE["mode"]
        _set("ESTOPPED")    # REAL VERSION: send safe-state command to the blimp
    return {"ok": True, "message": "Blimp stopped"}


@app.post("/api/resume")
def resume():
    _set(STATE["before_stop"] or "IDLE")
    return {"ok": True, "message": "Resumed"}


@app.post("/api/power_off")
def power_off():
    _set("IDLE")
    return {"ok": True, "message": "Blimp is off"}


@app.post("/api/setup/start")
def setup_start():
    STATE["map_pct"] = 0
    _set("MAPPING")
    return {"ok": True, "message": "Mapping started"}


@app.post("/api/setup/finish")
def setup_finish():
    _set("IDLE")
    return {"ok": True, "message": "Map saved"}


@app.get("/")
def home():
    return RedirectResponse("/driftblimp-ui.html")


@app.get("/driftblimp-ui.html")
def page():
    # Serve ONLY the page itself, so data/ and the source code are not browsable.
    return FileResponse(os.path.join(HERE, "driftblimp-ui.html"),
                        headers={"Cache-Control": "no-cache"})
