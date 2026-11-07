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
three retry attempts for training errors. After successful training,
`trainer.py` registers that run's `best.pt` in the local `active_model.txt`
file. Run `python vision.py` to load that model; the dataset path does not need
to be changed in `vision.py`.

To select a different existing checkpoint without changing code, edit
`active_model.txt` and set it to the checkpoint path relative to this
directory, for example:

```text
detect/elastic-robin2-9bb63ef8/weights/best.pt
```

`vision.py` loads the selected checkpoint and uses its class names. The file
path must point to an existing `.pt` model.

The camera script prints the trained model's class names and prompts for a
class name and minimum confidence. Matching is case-insensitive and uses the
model's exact class names. Only detections of the requested item at or above
the selected threshold are labeled while searching. The console reports
`Found <item> at <percent>% confidence` when the item first meets the
threshold. Press **O** to show or hide other recognized objects, and **Q** to
quit.
