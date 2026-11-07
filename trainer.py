#Input the dataset converted from convert_dataset.py into the code

from pathlib import Path
import shutil
import time

import torch
from ultralytics import YOLO

MAX_RETRIES = 3


def keep_recovery_checkpoint(trainer):
    last_checkpoint = Path(trainer.last)
    if not last_checkpoint.is_file():
        return

    backup_checkpoint = last_checkpoint.with_name('last-good.pt')
    temporary_checkpoint = backup_checkpoint.with_suffix('.pt.tmp')
    try:
        shutil.copyfile(last_checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(backup_checkpoint)
    except OSError as error:
        print(f'Warning: could not refresh recovery checkpoint: {error}')
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()


def find_interrupted_run(dataset_yaml, runs_directory):
    run_directories = [
        path for path in runs_directory.glob('train*')
        if path.is_dir()
    ]
    run_directories.sort(
        key=lambda path: max(
            (checkpoint.stat().st_mtime for checkpoint in (
                path / 'weights' / 'last.pt',
                path / 'weights' / 'last-good.pt',
            ) if checkpoint.is_file()),
            default=0,
        ),
        reverse=True,
    )

    for run_directory in run_directories:
        weights_directory = run_directory / 'weights'
        for checkpoint in (
            weights_directory / 'last.pt',
            weights_directory / 'last-good.pt',
        ):
            if not checkpoint.is_file():
                continue
            try:
                model = YOLO(str(checkpoint))
            except Exception as error:
                print(f'Cannot inspect checkpoint {checkpoint}: {error}')
                continue

            checkpoint_data = model.ckpt.get('train_args', {}).get('data', '')
            try:
                same_dataset = Path(checkpoint_data).resolve() == dataset_yaml.resolve()
            except (OSError, TypeError, ValueError):
                same_dataset = False

            epoch = model.ckpt.get('epoch', -1)
            has_optimizer = model.ckpt.get('optimizer') is not None
            train_args = model.ckpt.get('train_args', {})
            epochs = train_args.get('epochs', 0)
            stopper = model.ckpt.get('stopper') or {}
            best_epoch = stopper.get('best_epoch', epoch)
            patience = train_args.get('patience', epochs)
            valid_epoch = isinstance(epoch, int) and not isinstance(epoch, bool)
            already_stopped = (
                valid_epoch
                and isinstance(best_epoch, int)
                and isinstance(patience, int)
                and epoch - best_epoch >= patience
            )
            has_more_epochs = valid_epoch and isinstance(epochs, int) and epoch + 1 < epochs

            if same_dataset and has_optimizer and has_more_epochs and not already_stopped:
                return checkpoint, run_directory

    return None, None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA training was requested, but CUDA is unavailable in this Python '
            'environment. Install a CUDA-enabled PyTorch build and try again.'
        )

    print(f'Training on CUDA device: {torch.cuda.get_device_name(0)}')
    dataset_yaml = (
        Path(__file__).resolve().parent
        / 'DriftBlimp'
        / 'datasets'
        / 'elastic-robin-c5e850f5'
        / 'data.yaml'
    )
    runs_directory = Path(__file__).resolve().parent / 'runs' / 'detect'
    resume_checkpoint, run_directory = find_interrupted_run(dataset_yaml, runs_directory)
    if resume_checkpoint is None:
        yolo = YOLO('yolo11n.pt')
    else:
        print(f'Resuming unfinished run from {resume_checkpoint}')
        yolo = YOLO(str(resume_checkpoint))
    yolo.add_callback('on_model_save', keep_recovery_checkpoint)

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            if resume_checkpoint is None:
                yolo.train(data=str(dataset_yaml), epochs=100, imgsz=640, device=0)
            else:
                print(f'Resuming training from {resume_checkpoint}')
                yolo = YOLO(str(resume_checkpoint))
                yolo.add_callback('on_model_save', keep_recovery_checkpoint)
                yolo.train(
                    resume=str(resume_checkpoint),
                    data=str(dataset_yaml),
                    device=0,
                    save_dir=str(run_directory),
                )
            return
        except Exception as error:
            trainer = getattr(yolo, 'trainer', None)
            if trainer is not None:
                run_directory = Path(trainer.save_dir)
            if run_directory is None:
                raise RuntimeError(
                    'Training failed before a run directory was available; '
                    'there is no checkpoint to resume.'
                ) from error

            weights_directory = run_directory / 'weights'
            checkpoints = (
                weights_directory / 'last.pt',
                weights_directory / 'last-good.pt',
            )
            resume_checkpoint = None
            for checkpoint in checkpoints:
                if checkpoint.is_file():
                    try:
                        checkpoint_model = YOLO(str(checkpoint))
                    except Exception as checkpoint_error:
                        print(f'Cannot resume from {checkpoint}: {checkpoint_error}')
                    else:
                        if checkpoint_model.ckpt.get('optimizer') is not None:
                            resume_checkpoint = checkpoint
                            break
                        print(f'Checkpoint is not resumable: {checkpoint}')

            if resume_checkpoint is None:
                raise RuntimeError(
                    f'Training failed and no readable checkpoint was found in '
                    f'{weights_directory}. Latest error: {error}'
                ) from error
            if attempt > MAX_RETRIES:
                raise RuntimeError(
                    f'Training failed after {MAX_RETRIES} resume attempts. '
                    f'A readable checkpoint is available at {resume_checkpoint}. '
                    f'Latest error: {error}'
                ) from error

            print(
                f'Training attempt {attempt} failed: {error}\n'
                f'Retrying from {resume_checkpoint} in 5 seconds '
                f'({MAX_RETRIES - attempt + 1} retries remaining).'
            )
            time.sleep(5)


if __name__ == '__main__':
    main()