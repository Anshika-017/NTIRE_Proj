import os
import numpy as np

root = "data/train"

for scene in os.listdir(root):
    scene_path = os.path.join(root, scene)

    disp_path = os.path.join(scene_path, "disp_00.npy")

    if os.path.exists(disp_path):
        disp = np.load(disp_path)
        print(scene, "disp shape:", disp.shape)
        break
