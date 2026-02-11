import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class BoosterDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for scene in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene)

            cam_path = os.path.join(scene_path, "camera_00")
            disp_path = os.path.join(scene_path, "disp_00.npy")
            mask_path = os.path.join(scene_path, "mask_00.png")

            if not os.path.exists(disp_path):
                continue

            rgb_files = sorted(os.listdir(cam_path))

            for rgb_name in rgb_files:
                rgb_full = os.path.join(cam_path, rgb_name)

                self.samples.append({
                    "rgb": rgb_full,
                    "disp": disp_path,
                    "mask": mask_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["rgb"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load disparity
        disparity = np.load(sample["disp"]).astype("float32")

        # Load mask
        mask = cv2.imread(sample["mask"], 0)
        mask = (mask > 0).astype("float32")

        # Apply transforms
        if self.transform:
            augmented = self.transform(
                image=image,
                masks=[disparity, mask]
            )

            image = augmented["image"]
            disparity = augmented["masks"][0]
            mask = augmented["masks"][1]

        # Add channel dimension if needed
        if disparity.ndim == 2:
            disparity = disparity.unsqueeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, disparity, mask
