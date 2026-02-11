from dataset import BoosterDepthDataset
from transforms import get_train_transforms

dataset = BoosterDepthDataset("data/train", transform=get_train_transforms())

print("Total samples:", len(dataset))

img, disp, mask = dataset[0]
print(img.shape, disp.shape, mask.shape)
