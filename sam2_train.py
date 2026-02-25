import sys
import torch
import cv2
from pathlib import Path

# ----------------------------
# ADD SAM2 TO PATH
# ----------------------------
sys.path.insert(0, "../sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ----------------------------
# PATHS
# ----------------------------
CONFIG = "sam2_hiera_l"
CHECKPOINT = "../checkpoints/sam2_hiera_large.pt"

DATASET = Path("../train")
OUTPUT = Path("../outputs")
OUTPUT.mkdir(exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = build_sam2(
    config_file=CONFIG,
    ckpt_path=CHECKPOINT,
    device=DEVICE,
)

predictor = SAM2ImagePredictor(model)

# ----------------------------
# COLLECT IMAGES
# ----------------------------
def collect_images(root):
    images = []
    for p in root.rglob("*.png"):
        if "camera_" in str(p):
            images.append(p)
    return images


images = collect_images(DATASET)
print("Total images:", len(images))

# ----------------------------
# TRAIN LOOP
# ----------------------------
model.train()

for img_path in images:

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        multimask_output=True,
    )

    best_mask = masks[scores.argmax()]
    best_mask = (best_mask * 255).astype("uint8")

    save_path = OUTPUT / img_path.name
    cv2.imwrite(str(save_path), best_mask)

    print("Processed:", img_path.name)

# ----------------------------
# SAVE FINETUNED MODEL
# ----------------------------
torch.save(model.state_dict(), "../checkpoints/sam2_finetuned.pt")

print("Training Finished")