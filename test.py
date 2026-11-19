import argparse
import os
from pathlib import Path
from time import perf_counter

# Ollama should not try to load this workstation's GTX 970 with a newer PTX
# toolchain. num_gpu=0 below also disables GPU layers for this request.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OLLAMA_NUM_GPU'] = '0'

from ollama import ChatResponse, ResponseError, chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Describe an image with a local Ollama vision model.'
    )
    parser.add_argument(
        'image',
        nargs='?',
        help='Path to the image.',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'gemma4'),
        help='Ollama model name (default: gemma4).',
    )
    return parser.parse_args()


def get_image_path(args: argparse.Namespace) -> Path:
    path = args.image
    if path is None:
        path = input('Please enter the path to the image: ').strip()

    path = path.strip().strip('"')
    image_path = Path(path).expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f'Image file not found: {image_path}')

    return image_path



def main() -> None:
    start = perf_counter()
    args = parse_args()
    image_path = get_image_path(args)

    try:
        response: ChatResponse = chat(
            model=args.model,
            options={
                'num_gpu': 0,
            },
            messages=[
                {
                    'role': 'user',
                    'content': 'If there are eyeglasses, describe the eyeglasses concisely in one sentence 20 words.',
                    'images': [str(image_path)],
                }
            ],
        )
    except (ResponseError, ConnectionError) as error:
        raise SystemExit(f'Ollama request failed: {error}')

    print(response.message.content)
    end=perf_counter()
    print(end-start)


if __name__ == '__main__':
    main()