"""
Ultralytics models.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import ultralytics

import eta.core.web as etaw

from fiftyone.operators import types
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


def load_model(model_name, model_path, classes=None):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        classes (None): an optional list of classes to use for zero-shot
            prediction

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    model_type = MODEL_TYPES[model_name]

    if model_type == "rtdetr":
        model = ultralytics.RTDETR(model=model_path)
    else:
        model = ultralytics.YOLO(model=model_path)

    if model_type == "classification":
        output_processor_cls = fouu.UltralyticsClassificationOutputProcessor
    elif model_type == "detection":
        output_processor_cls = fouu.UltralyticsDetectionOutputProcessor
    elif model_type == "segmentation":
        output_processor_cls = fouu.UltralyticsSegmentationOutputProcessor
    elif model_type == "pose":
        output_processor_cls = fouu.UltralyticsPoseOutputProcessor
    elif model_type == "obb":
        output_processor_cls = fouu.UltralyticsOBBOutputProcessor
    elif model_type == "rtdetr":
        output_processor_cls = fouu.UltralyticsDetectionOutputProcessor

    return fouu.FiftyOneYOLOModel(
        fouu.FiftyOneYOLOModelConfig(
            dict(
                model=model,
                model_path=model_path,
                output_processor_cls=output_processor_cls,
                classes=classes,
            )
        )
    )


def resolve_input(model_name, ctx):
    """Defines any necessary properties to collect the model's custom
    parameters from a user during prompting.

    Args:
        model_name: the name of the model, as declared by the ``base_name`` and
            optional ``version`` fields of the manifest
        ctx: an :class:`fiftyone.operators.ExecutionContext`

    Returns:
        a :class:`fiftyone.operators.types.Property`, or None
    """
    if model_name not in ZERO_SHOT_MODELS:
        return

    inputs = types.Object()

    inputs.list(
        "classes",
        types.String(),
        required=False,
        default=None,
        label="Zero shot classes",
        description=(
            "An optional list of custom classes for zero-shot prediction"
        ),
        view=types.AutocompleteView(),
    )

    return types.Property(inputs)


ZERO_SHOT_MODELS = {
    # YOLO8
    "voxel51/yolov8s-world-torch",
    "voxel51/yolov8m-world-torch",
    "voxel51/yolov8l-world-torch",
    "voxel51/yolov8x-world-torch",

    # YOLOEv8
    "voxel51/yoloev8s-seg-torch",
    "voxel51/yoloev8m-seg-torch",
    "voxel51/yoloev8l-seg-torch",

    # YOLOE11
    "voxel51/yoloe11s-seg-torch",
    "voxel51/yoloe11m-seg-torch",
    "voxel51/yoloe11l-seg-torch",
}

MODEL_URLS = {
    # YOLOv8
    "voxel51/yolov8n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "voxel51/yolov8s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    "voxel51/yolov8m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    "voxel51/yolov8l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "voxel51/yolov8x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
    "voxel51/yolov8n-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt",
    "voxel51/yolov8s-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt",
    "voxel51/yolov8m-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt",
    "voxel51/yolov8l-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt",
    "voxel51/yolov8x-oiv7-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt",
    "voxel51/yolov8n-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt",
    "voxel51/yolov8s-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt",
    "voxel51/yolov8m-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt",
    "voxel51/yolov8l-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt",
    "voxel51/yolov8x-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt",
    "voxel51/yolov8s-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt",
    "voxel51/yolov8m-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-world.pt",
    "voxel51/yolov8l-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt",
    "voxel51/yolov8x-world-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-world.pt",
    "voxel51/yolov8n-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt",
    "voxel51/yolov8s-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-obb.pt",
    "voxel51/yolov8m-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-obb.pt",
    "voxel51/yolov8l-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-obb.pt",
    "voxel51/yolov8x-obb-dotav1-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-obb.pt",

    # YOLOEv8
    "voxel51/yoloev8s-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8s-seg.pt",
    "voxel51/yoloev8m-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8m-seg.pt",
    "voxel51/yoloev8l-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt",

    # YOLOv9
    "voxel51/yolov9c-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c.pt",
    "voxel51/yolov9e-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e.pt",
    "voxel51/yolov9c-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c-seg.pt",
    "voxel51/yolov9e-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e-seg.pt",

    # YOLOv10
    "voxel51/yolov10n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
    "voxel51/yolov10s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
    "voxel51/yolov10m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt",
    "voxel51/yolov10l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt",
    "voxel51/yolov10x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt",

    # RTDETR
    "voxel51/rtdetr-l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt",
    "voxel51/rtdetr-x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt",

    # YOLO11
    "voxel51/yolo11n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "voxel51/yolo11s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "voxel51/yolo11m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "voxel51/yolo11l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "voxel51/yolo11x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    "voxel51/yolo11n-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    "voxel51/yolo11s-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
    "voxel51/yolo11m-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
    "voxel51/yolo11l-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
    "voxel51/yolo11x-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",

    # YOLOE11
    "voxel51/yoloe11s-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt",
    "voxel51/yoloe11m-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg.pt",
    "voxel51/yoloe11l-seg-torch": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt",

    # YOLO26
    "voxel51/yolo26n-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "voxel51/yolo26s-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "voxel51/yolo26m-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "voxel51/yolo26l-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "voxel51/yolo26x-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    "voxel51/yolo26n-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
    "voxel51/yolo26s-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
    "voxel51/yolo26m-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
    "voxel51/yolo26l-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt",
    "voxel51/yolo26x-seg-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
    "voxel51/yolo26n-cls-imagenet-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt",
    "voxel51/yolo26s-cls-imagenet-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-cls.pt",
    "voxel51/yolo26m-cls-imagenet-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt",
    "voxel51/yolo26l-cls-imagenet-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-cls.pt",
    "voxel51/yolo26x-cls-imagenet-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt",
    "voxel51/yolo26n-pose-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
    "voxel51/yolo26s-pose-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
    "voxel51/yolo26m-pose-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt",
    "voxel51/yolo26l-pose-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt",
    "voxel51/yolo26x-pose-coco-torch": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
}

MODEL_TYPES = {
    # YOLOv8
    "voxel51/yolov8n-coco-torch": "detection",
    "voxel51/yolov8s-coco-torch": "detection",
    "voxel51/yolov8m-coco-torch": "detection",
    "voxel51/yolov8l-coco-torch": "detection",
    "voxel51/yolov8x-coco-torch": "detection",
    "voxel51/yolov8n-oiv7-torch": "detection",
    "voxel51/yolov8s-oiv7-torch": "detection",
    "voxel51/yolov8m-oiv7-torch": "detection",
    "voxel51/yolov8l-oiv7-torch": "detection",
    "voxel51/yolov8x-oiv7-torch": "detection",
    "voxel51/yolov8n-seg-coco-torch": "segmentation",
    "voxel51/yolov8s-seg-coco-torch": "segmentation",
    "voxel51/yolov8m-seg-coco-torch": "segmentation",
    "voxel51/yolov8l-seg-coco-torch": "segmentation",
    "voxel51/yolov8x-seg-coco-torch": "segmentation",
    "voxel51/yolov8s-world-torch": "detection",
    "voxel51/yolov8m-world-torch": "detection",
    "voxel51/yolov8l-world-torch": "detection",
    "voxel51/yolov8x-world-torch": "detection",
    "voxel51/yolov8n-obb-dotav1-torch": "obb",
    "voxel51/yolov8s-obb-dotav1-torch": "obb",
    "voxel51/yolov8m-obb-dotav1-torch": "obb",
    "voxel51/yolov8l-obb-dotav1-torch": "obb",
    "voxel51/yolov8x-obb-dotav1-torch": "obb",

    # YOLOEv8
    "voxel51/yoloev8s-seg-torch": "segmentation",
    "voxel51/yoloev8m-seg-torch": "segmentation",
    "voxel51/yoloev8l-seg-torch": "segmentation",

    # YOLOv9
    "voxel51/yolov9c-coco-torch": "detection",
    "voxel51/yolov9e-coco-torch": "detection",
    "voxel51/yolov9c-seg-coco-torch": "segmentation",
    "voxel51/yolov9e-seg-coco-torch": "segmentation",

    # YOLOv10
    "voxel51/yolov10n-coco-torch": "detection",
    "voxel51/yolov10s-coco-torch": "detection",
    "voxel51/yolov10m-coco-torch": "detection",
    "voxel51/yolov10l-coco-torch": "detection",
    "voxel51/yolov10x-coco-torch": "detection",

    # RTDETR
    "voxel51/rtdetr-l-coco-torch": "rtdetr",
    "voxel51/rtdetr-x-coco-torch": "rtdetr",

    # YOLO11
    "voxel51/yolo11n-coco-torch": "detection",
    "voxel51/yolo11s-coco-torch": "detection",
    "voxel51/yolo11m-coco-torch": "detection",
    "voxel51/yolo11l-coco-torch": "detection",
    "voxel51/yolo11x-coco-torch": "detection",
    "voxel51/yolo11n-seg-coco-torch": "segmentation",
    "voxel51/yolo11s-seg-coco-torch": "segmentation",
    "voxel51/yolo11m-seg-coco-torch": "segmentation",
    "voxel51/yolo11l-seg-coco-torch": "segmentation",
    "voxel51/yolo11x-seg-coco-torch": "segmentation",

    # YOLOE11
    "voxel51/yoloe11s-seg-torch": "segmentation",
    "voxel51/yoloe11m-seg-torch": "segmentation",
    "voxel51/yoloe11l-seg-torch": "segmentation",

    # YOLO26
    "voxel51/yolo26n-coco-torch": "detection",
    "voxel51/yolo26s-coco-torch": "detection",
    "voxel51/yolo26m-coco-torch": "detection",
    "voxel51/yolo26l-coco-torch": "detection",
    "voxel51/yolo26x-coco-torch": "detection",
    "voxel51/yolo26n-seg-coco-torch": "segmentation",
    "voxel51/yolo26s-seg-coco-torch": "segmentation",
    "voxel51/yolo26m-seg-coco-torch": "segmentation",
    "voxel51/yolo26l-seg-coco-torch": "segmentation",
    "voxel51/yolo26x-seg-coco-torch": "segmentation",
    "voxel51/yolo26n-cls-imagenet-torch": "classification",
    "voxel51/yolo26s-cls-imagenet-torch": "classification",
    "voxel51/yolo26m-cls-imagenet-torch": "classification",
    "voxel51/yolo26l-cls-imagenet-torch": "classification",
    "voxel51/yolo26x-cls-imagenet-torch": "classification",
    "voxel51/yolo26n-pose-coco-torch": "pose",
    "voxel51/yolo26s-pose-coco-torch": "pose",
    "voxel51/yolo26m-pose-coco-torch": "pose",
    "voxel51/yolo26l-pose-coco-torch": "pose",
    "voxel51/yolo26x-pose-coco-torch": "pose",
}
