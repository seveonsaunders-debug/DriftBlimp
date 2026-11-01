# DriftBlimp
Code for the DriftBlimp Home base Sub-System

## Camera object recognition

Run `python vision.py`. The script uses the YOLO11n COCO model and prompts for
an item name and a minimum confidence percentage. Enter a YOLO class name (for
example, `bottle`); the supported class names are printed before the prompt.
Matching is case-insensitive and uses the model's exact class names.

Only detections of the requested item at or above the selected threshold are
labeled while searching. The console reports `Found <item> at <percent>%
confidence` when the item first meets the threshold. Press **O** to show or
hide other recognized objects, and **Q** to quit. The initial setting hides
other objects so the requested item stays in focus.
