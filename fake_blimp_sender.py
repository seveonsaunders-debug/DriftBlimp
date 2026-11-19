"""
Fake blimp sender, for testing WITHOUT a camera, Arduino, or blimp.

It does what blimp_sender_test.py does (POST JPEG photos to the receiver's
/upload every couple of seconds, then watch /status), but the photos come
from a folder of saved pictures instead of a live camera.

Setup:
  1. Make a folder called test_images next to this file.
  2. Put some .jpg photos in it. Good choices:
       - Pictures from the training dataset, e.g.
         datasets\\<dataset-name>\\images\\train\\  (copy 5-10 of them)
       - Your own photos of keys, a phone, a bottle, etc. (under 2 MB each)
     At least one photo must clearly show the item you search for.
  3. Start homebase_reciever.py, then run:   python fake_blimp_sender.py
     Optional: python fake_blimp_sender.py C:\\path\\to\\some\\folder

Press Ctrl+C to stop.
"""
import sys
import time
from pathlib import Path

import requests

HOMEBASE_URL = "http://localhost:5000"      # the receiver (same computer)
IMAGE_FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "test_images"
SEND_INTERVAL = 2.0                         # seconds between photos
MAX_BYTES = 2 * 1024 * 1024                 # the receiver rejects anything larger


def find_images(folder):
    files = sorted(p for p in folder.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file())
    return [p for p in files if p.stat().st_size <= MAX_BYTES]


def main():
    images = find_images(IMAGE_FOLDER)
    if not images:
        print(f"No usable .jpg photos found in: {IMAGE_FOLDER}")
        print("Put some JPEG photos (under 2 MB each) there and run again.")
        return

    print(f"Found {len(images)} photos in {IMAGE_FOLDER}")
    print(f"Sending one every {SEND_INTERVAL:g} s to {HOMEBASE_URL}. Ctrl+C to stop.\n")

    session = requests.Session()
    last_event = None
    n = 0
    try:
        while True:
            path = images[n % len(images)]
            n += 1
            try:
                r = session.post(HOMEBASE_URL + "/upload",
                                 data=path.read_bytes(),
                                 headers={"Content-Type": "image/jpeg"},
                                 timeout=5)
                print(f"Sent {path.name}: {'ok' if r.ok else 'FAILED ' + str(r.status_code)}")

                data = session.get(HOMEBASE_URL + "/status", timeout=2).json()
                event = int(data["event_id"])
                if last_event is None:
                    last_event = event
                elif event > last_event:
                    last_event = event
                    print(f"  >>> TARGET FOUND (event {event}) in {path.name}")
            except (requests.RequestException, ValueError, KeyError) as error:
                print("Problem talking to the receiver:", error)

            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
