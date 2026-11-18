# Convert NDJSON dataset to YOLO format from Ultralytics

import asyncio
from ultralytics.data.converter import convert_ndjson_to_yolo

yaml_path = asyncio.run(
    convert_ndjson_to_yolo(r"C:\Users\Administrator\DriftBlimp-2\glasses.ndjson")
)
print(f"Dataset YAML: {yaml_path}")