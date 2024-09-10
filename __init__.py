"""
Ultralytics models.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import eta.core.web as etaw

import fiftyone.utils.ultralytics as fouu


def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    url = MODEL_URLS[model_name]
    etaw.download_file(url, path=model_path)


def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    model_type = MODEL_TYPES[model_name]

    if model_type == "detection":
        config = fouu.FiftyOneYOLODetectionModelConfig(
            dict(model_path=model_path)
        )
        return fouu.FiftyOneYOLODetectionModel(config)

    if model_type == "segmentation":
        config = fouu.FiftyOneYOLOSegmentationModelConfig(
            dict(model_path=model_path)
        )
        return fouu.FiftyOneYOLOSegmentationModel(config)

    if model_type == "rtdetr":
        config = fouu.FiftyOneRTDETRModelConfig(
            dict(model_path=model_path)
        )
        return fouu.FiftyOneRTDETRModel(config)

    if model_type == "obb":
        config = fouu.FiftyOneYOLOOBBModelConfig(
            dict(model_path=model_path)
        )
        return fouu.FiftyOneYOLOOBBModel(config)


MODEL_URLS = {
    "voxel51/yolov8n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "voxel51/yolov8s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    "voxel51/yolov8m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    "voxel51/yolov8l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "voxel51/yolov8x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
    "voxel51/yolov8n-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt",
    "voxel51/yolov8s-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt",
    "voxel51/yolov8m-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt",
    "voxel51/yolov8l-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt",
    "voxel51/yolov8x-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt",
    "voxel51/yolov9c-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c.pt",
    "voxel51/yolov9e-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e.pt",
    "voxel51/yolov9c-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c-seg.pt",
    "voxel51/yolov9e-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e-seg.pt",
    "voxel51/yolov10n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
    "voxel51/yolov10s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
    "voxel51/yolov10m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt",
    "voxel51/yolov10l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt",
    "voxel51/yolov10x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt",
    "voxel51/rtdetr-l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt",
    "voxel51/rtdetr-x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt",
    "voxel51/yolov8s-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt",
    "voxel51/yolov8m-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-world.pt",
    "voxel51/yolov8l-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt",
    "voxel51/yolov8x-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-world.pt",
    "voxel51/yolov8n-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt",
    "voxel51/yolov8s-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-obb.pt",
    "voxel51/yolov8m-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-obb.pt",
    "voxel51/yolov8l-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-obb.pt",
    "voxel51/yolov8x-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-obb.pt",
    "voxel51/yolov8n-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt",
    "voxel51/yolov8s-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt",
    "voxel51/yolov8m-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt",
    "voxel51/yolov8l-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt",
    "voxel51/yolov8x-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt",
}

MODEL_TYPES = {
    "voxel51/yolov8n-coco-torch": "detection",
    "voxel51/yolov8s-coco-torch": "detection",
    "voxel51/yolov8m-coco-torch": "detection",
    "voxel51/yolov8l-coco-torch": "detection",
    "voxel51/yolov8x-coco-torch": "detection",
    "voxel51/yolov8n-seg-coco-torch": "detection",
    "voxel51/yolov8s-seg-coco-torch": "detection",
    "voxel51/yolov8m-seg-coco-torch": "detection",
    "voxel51/yolov8l-seg-coco-torch": "detection",
    "voxel51/yolov8x-seg-coco-torch": "detection",
    "voxel51/yolov9c-coco-torch": "detection",
    "voxel51/yolov9e-coco-torch": "detection",
    "voxel51/yolov9c-seg-coco-torch": "segmentation",
    "voxel51/yolov9e-seg-coco-torch": "segmentation",
    "voxel51/yolov10n-coco-torch": "detection",
    "voxel51/yolov10s-coco-torch": "detection",
    "voxel51/yolov10m-coco-torch": "detection",
    "voxel51/yolov10l-coco-torch": "detection",
    "voxel51/yolov10x-coco-torch": "detection",
    "voxel51/rtdetr-l-coco-torch": "rtdetr",
    "voxel51/rtdetr-x-coco-torch": "rtdetr",
    "voxel51/yolov8s-world-torch": "detection",
    "voxel51/yolov8m-world-torch": "detection",
    "voxel51/yolov8l-world-torch": "detection",
    "voxel51/yolov8x-world-torch": "detection",
    "voxel51/yolov8n-obb-dotav1-torch": "obb",
    "voxel51/yolov8s-obb-dotav1-torch": "obb",
    "voxel51/yolov8m-obb-dotav1-torch": "obb",
    "voxel51/yolov8l-obb-dotav1-torch": "obb",
    "voxel51/yolov8x-obb-dotav1-torch": "obb",
    "voxel51/yolov8n-oiv7-torch": "detection",
    "voxel51/yolov8s-oiv7-torch": "detection",
    "voxel51/yolov8m-oiv7-torch": "detection",
    "voxel51/yolov8l-oiv7-torch": "detection",
    "voxel51/yolov8x-oiv7-torch": "detection",
}
