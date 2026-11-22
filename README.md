# DriftBlimp
Code for the DriftBlimp Home base Sub-System

## Train and use an object detector

In `trainer.py`, set `DATASET_YAML` to the `data.yaml` for the dataset you want
to train, for example:

```python
DATASET_YAML = PROJECT_DIRECTORY / 'datasets' / 'elastic-robin2-9bb63ef8' / 'data.yaml'
```

From this directory, run:

```powershell
python trainer.py
```

Training uses CUDA device 0 and saves run results under `detect/`. An
interrupted run for the same dataset is resumed automatically, with up to
three retry attempts for training errors.

Run `vision.py` with the trained `.pt` file as a path relative to the current
directory:

```powershell
python vision.py detect/elastic-robin2-9bb63ef8/weights/best.pt
```

For example, if you are in `DriftBlimp` and the model is `glasses.pt`, run:

```powershell
python vision.py glasses.pt
```

You can also run `python vision.py` without an argument and enter the model
path at the prompt. Relative paths are resolved from the directory where the
command is run.

The camera script prints the trained model's class names and prompts for a
class name and minimum confidence. Matching is case-insensitive and uses the
model's exact class names. Only detections of the requested item at or above
the selected threshold are labeled while searching. The console reports
`Found <item> at <percent>% confidence` when the item first meets the
threshold. Press **O** to show or hide other recognized objects, and **Q** to
quit.
