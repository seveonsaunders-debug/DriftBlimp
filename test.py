from ollama import chat
from pathlib import Path

# Pass in the path to the image
path = input('Please enter the path to the image: ').strip().strip('"')
image_path = Path(path).expanduser()

if not image_path.is_file():
    raise FileNotFoundError(f'Image file not found: {image_path}')

# You can also pass in base64 encoded image data
# img = base64.b64encode(image_path.read_bytes()).decode()
# or the raw bytes
# img = image_path.read_bytes()

response = chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image? Be concise.',
      'images': [str(image_path.resolve())],
    }
  ],
  options={'num_gpu': 0},
)

print(response.message.content)