"""
DriftBlimp UI server
Sits between the iPad app and the home base receiver (homebase_reciever.py).

MOCK mode (default): pretends to be the home base. No blimp needed.
REAL mode: forwards searches to the real receiver on port 5000.
           Needs the receiver to have GET /classes and POST /target.

Run from the ui folder:
    pip install fastapi uvicorn requests
    python -m uvicorn ui_server:app --host 0.0.0.0 --port 8000

Real mode (PowerShell):
    $env:DRIFT_MODE="real"
    python -m uvicorn ui_server:app --host 0.0.0.0 --port 8000
"""
import os
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODE = os.environ.get("DRIFT_MODE", "mock").lower()          # "mock" or "real"
HB = os.environ.get("HOMEBASE_URL", "http://localhost:5000")  # the real receiver
HERE = os.path.dirname(os.path.abspath(__file__))
STEP_SECONDS = 2.5
MOCK_CLASSES = ["keys", "glasses", "phone", "tv remote", "pill box"]

app = FastAPI()

# mode: IDLE, SEARCHING, RETURNING, DOCKED, ESTOPPED, MAPPING
STATE = {"mode": "IDLE", "item_id": None, "t0": 0.0, "before_stop": None,
         "map_pct": 0, "battery": 85, "found": False, "baseline": 0,
         "hb_ok": True}


class SearchRequest(BaseModel):
    item_id: int
    name: str = ""
    confidence: float = 0.5


def _elapsed():
    return time.time() - STATE["t0"]


def _set(mode):
    STATE["mode"] = mode
    STATE["t0"] = time.time()


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
                    STATE["found"] = True
                    _set("RETURNING")
                    mode, stage = "RETURNING", 0
            except (HTTPException, ValueError, KeyError):
                pass
        else:                                                  # MOCK
            stage = int(_elapsed() // STEP_SECONDS)
            if stage >= 3:
                STATE["found"] = True
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
    """Names the home base knows how to find (for the Add Item screen)."""
    if MODE == "real":
        try:
            return _homebase("GET", "/classes").json()
        except (HTTPException, ValueError):
            return []
    return MOCK_CLASSES


@app.post("/api/search")
def search(req: SearchRequest):
    if MODE == "real":
        r = _homebase("POST", "/target",
                      json={"name": req.name, "confidence": req.confidence})
        if r.status_code == 400:
            raise HTTPException(
                400, f"The home base doesn't know how to find \"{req.name}\" yet.")
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


# Serves driftblimp-ui.html. Must stay LAST so it doesn't hide the /api routes.
app.mount("/", StaticFiles(directory=HERE), name="ui")
