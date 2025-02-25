import os
from typing import ClassVar

import supervision as sv
from PIL import Image
from torch.utils.data import Dataset

from maestro.trainer.common.datasets.base import BaseDetectionDataset
from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


class YOLODataset(Dataset, BaseDetectionDataset):
    """
    A dataset for loading images and YOLO-format annotations from a directory of label files.

    This class reads annotation entries from YOLO text files and ensures that each image
    referenced in the annotations exists in the supplied image directory. It leverages the YOLO
    format to extract the list of classes, image file paths, and associated detection annotations.
    Entries that fail validation (for example, due to missing image files) are skipped with
    appropriate warnings logged.

    Parameters:
        data_yaml_path (str): Filesystem path to the YOLO data YAML file.
        images_directory_path (str): Filesystem path to the directory where image files are stored.
        annotations_directory_path (str): Filesystem path to the directory containing YOLO label files.

    Example:
        ```
        from roboflow import download_dataset, login
        from maestro.trainer.common.datasets.yolo import YOLODataset

        login()

        dataset = download_dataset("universe.roboflow.com/username/project-name/1", "yolov5")
        ds = YOLODataset(
            data_yaml_path=f"{dataset.location}/data.yaml",
            images_directory_path=f"{dataset.location}/images/test",
            annotations_directory_path=f"{dataset.location}/labels/test"
        )
        len(ds)
        # 430
        ```
    """

    ROBOFLOW_YOLO_DATA_YAML: ClassVar[str] = "data.yaml"
    REQUIRED_KEYS: ClassVar[list[str]] = ["train", "val", "names"]

    def __init__(self, data_yaml_path: str, images_directory_path: str, annotations_directory_path: str) -> None:
        self.images_directory_path = images_directory_path
        self.classes, self.entries = self._load_entries(
            data_yaml_path=data_yaml_path,
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
        )

    @classmethod
    def _load_entries(
        cls, data_yaml_path: str, images_directory_path: str, annotations_directory_path: str
    ) -> tuple[list[str], list[tuple[str, sv.Detections]]]:
        """
        Load and parse YOLO annotations.

        Returns:
            Tuple:
                A tuple containing:
                - A list of class names.
                - A list of valid entries, where each entry is a tuple containing the image path
                  and the corresponding detections.
        """
        if not os.path.isfile(data_yaml_path):
            logger.warning(f"Data YAML file does not exist: '{data_yaml_path}'")
            return [], []

        try:
            classes, image_paths, annotation_list = sv.load_yolo_annotations(
                images_directory_path=images_directory_path,
                annotations_directory_path=annotations_directory_path,
                data_yaml_path=data_yaml_path,
                force_masks=False,
                is_obb=False,
            )
        except Exception as e:
            logger.warning(f"Could not parse YOLO annotations from '{data_yaml_path}': {e}")
            return [], []

        total_images = len(image_paths)
        skipped_count = 0
        empty_detections_count = 0
        valid_entries: list[tuple[str, sv.Detections]] = []

        for image_path, detections in zip(image_paths, annotation_list):
            if not os.path.exists(image_path):
                skipped_count += 1
                logger.warning(f"Skipping file: image file not found '{image_path}'")
                continue

            if detections.xyxy.shape[0] == 0:
                empty_detections_count += 1

            valid_entries.append((image_path, detections))

        loaded_count = total_images - skipped_count
        if total_images > 0:
            logger.info(
                f"Loaded {loaded_count} valid entries out of {total_images} from '{data_yaml_path}'. "
                f"Skipped {skipped_count}. Found {empty_detections_count} entries with empty detections."
            )
        else:
            logger.warning(f"No images found in '{data_yaml_path}'.")

        return classes, valid_entries

    def __len__(self) -> int:
        """
        Return the number of valid entries in the dataset.

        Returns:
            int: Total count of dataset entries.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, sv.Detections]:
        """
        Retrieve the image and its corresponding annotation entry at the specified index.

        Parameters:
            idx (int): The zero-based index of the desired entry.

        Returns:
            tuple: A tuple containing:
                - PIL.Image.Image: The image object.
                - sv.Detections: The corresponding annotation entry.

        Raises:
            IndexError: If the index is out of the valid range.
        """
        if idx >= len(self.entries):
            raise IndexError(f"Index {idx} is out of range.")
        image_path, detections = self.entries[idx]
        image = Image.open(image_path).convert("RGB")
        return image, detections
