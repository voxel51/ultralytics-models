# Ultralytics Models

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
