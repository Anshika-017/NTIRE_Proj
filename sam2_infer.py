import sys
import torch
import cv2

sys.path.insert(0, "../sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam2(
    config_file="sam2_hiera_l",
    ckpt_path="../checkpoints/sam2_finetuned.pt",
    device=DEVICE,
)

predictor = SAM2ImagePredictor(model)

image = cv2.imread("../train/Bathroom/camera_00/im0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

masks, scores, _ = predictor.predict()

mask = masks[scores.argmax()] * 255

cv2.imwrite("../outputs/test_result.png", mask)

print("Inference Done")