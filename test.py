import argparse
import os
from pathlib import Path

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
    args = parse_args()
    image_path = get_image_path(args)

    try:
        response: ChatResponse = chat(
            model=args.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'What is in this image? Be concise.',
                    'images': [str(image_path)],
                }
            ],
        )
    except (ResponseError, ConnectionError) as error:
        raise SystemExit(f'Ollama request failed: {error}')

    print(response.message.content)


if __name__ == '__main__':
    main()