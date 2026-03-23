# Ultralytics Models

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

Wrapper for various [Ultralytics models](https://www.ultralytics.com) for the
FiftyOne Model Zoo.

## Example usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=50,
    shuffle=True,
)

foz.register_zoo_model_source("https://github.com/voxel51/ultralytics-models")
model = foz.load_zoo_model("voxel51/yolov10s-coco-torch")

dataset.apply_model(model, label_field="predictions")
```

## Test all models

```py
import fiftyone as fo
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/voxel51/ultralytics-models")

# All models
# YOLO_MODELS = foz.list_zoo_models(source="https://github.com/voxel51/ultralytics-models")

# Only smallest version of each model
YOLO_MODELS = [
    # YOLOv8
    "voxel51/yolov8n-coco-torch",
    "voxel51/yolov8n-oiv7-torch",
    "voxel51/yolov8n-seg-coco-torch",
    "voxel51/yolov8s-world-torch",  # zero shot
    "voxel51/yolov8n-obb-dotav1-torch",

    # YOLOEv8
    "voxel51/yoloev8s-seg-torch",  # zero shot

    # YOLOv9
    "voxel51/yolov9c-coco-torch",
    "voxel51/yolov9c-seg-coco-torch",

    # YOLOv10
    "voxel51/yolov10n-coco-torch",

    # RTDETR
    "voxel51/rtdetr-l-coco-torch",

    # YOLO11
    "voxel51/yolo11n-coco-torch",
    "voxel51/yolo11n-seg-coco-torch",

    # YOLOE11
    "voxel51/yoloe11s-seg-torch",  # zero shot

    # YOLO26
    "voxel51/yolo26n-coco-torch",
    "voxel51/yolo26n-seg-coco-torch",
    "voxel51/yolo26n-cls-imagenet-torch",
    "voxel51/yolo26n-pose-coco-torch",
]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=10,
    shuffle=True,
)

# For zero-shot models
classes = ["person", "dog", "cat", "bird", "car", "tree", "chair"]

for model_name in YOLO_MODELS:
    print(f"Testing '{model_name}'")

    zoo_model = foz.get_zoo_model(model_name)
    if "zero-shot" in zoo_model.tags:
        model = foz.load_zoo_model(model_name, classes=classes)
    else:
        model = foz.load_zoo_model(model_name)

    label_field = model_name.split("/")[-1].replace("-", "_")
    dataset.apply_model(model, label_field=label_field)

session = fo.launch_app(dataset)
```
