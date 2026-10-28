import fiftyone.zoo as foz
import fiftyone.types as fot

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Mobile phone"],
    max_samples=800   # enough for quick training
)

dataset.export(
    export_dir="phone_dataset",
    dataset_type=fot.YOLOv5Dataset
)