import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose(
        [
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ],
        additional_targets={"mask2": "mask"}   # ← THIS IS IMPORTANT
    )

def get_val_transforms():
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ],
        additional_targets={"mask2": "mask"}   # ← ALSO HERE
    )
