import argparse
import os

import torch
import numpy as np
import glob
from pointcept.utils.logger import get_root_logger

CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]


def evaluation(intersection, union, target, logger=None):
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    if logger is not None:
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(len(CLASS_NAMES)):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=CLASS_NAMES[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record_root",
        required=True,
        help="Path to the S3DIS record of each split",
    )
    config = parser.parse_args()
    logger = get_root_logger(
        log_file=os.path.join(config.record_root, "6-fold.log"),
        file_mode="w",
    )

    records = sorted(glob.glob(os.path.join(config.record_root, "Area_*.pth")))
    assert len(records) == 6
    intersection_ = np.zeros(len(CLASS_NAMES), dtype=int)
    union_ = np.zeros(len(CLASS_NAMES), dtype=int)
    target_ = np.zeros(len(CLASS_NAMES), dtype=int)

    for record in records:
        area = os.path.basename(record).split(".")[0]
        info = torch.load(record)
        logger.info(f"<<<<<<<<<<<<<<<<< Parsing {area} <<<<<<<<<<<<<<<<<")
        intersection = info["intersection"]
        union = info["union"]
        target = info["target"]
        evaluation(intersection, union, target, logger=logger)
        intersection_ += intersection
        union_ += union
        target_ += target

    logger.info(f"<<<<<<<<<<<<<<<<< Parsing 6-fold <<<<<<<<<<<<<<<<<")
    evaluation(intersection_, union_, target_, logger=logger)


if __name__ == "__main__":
    main()
