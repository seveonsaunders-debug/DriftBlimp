# Convert NDJSON dataset to YOLO format from Ultralytics

import asyncio
from ultralytics.data.converter import convert_ndjson_to_yolo

yaml_path = asyncio.run(
    convert_ndjson_to_yolo(r"C:\Users\Administrator\Desktop\DriftBlimp\DriftBlimp\peaceful-heron.ndjson", output_path="datasets")
)
print(f"Dataset YAML: {yaml_path}")