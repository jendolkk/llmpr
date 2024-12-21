from PIL import Image
from torch.utils.data import Dataset
import os

class DP2Data(Dataset):
  def __init__(self, image_dir, json_file, transform=None):
    self.image_dir = image_dir
    self.image_list = os.listdir(image_dir)
    import json
    with open(json_file, 'r') as f:
      self.dicts = json.load(f)
    self.transform = transform

  def __len__(self):
    return len(self.dicts)

  def __getitem__(self, idx):
    mp = self.dicts[idx]
    image = Image.open(os.path.join(self.image_dir, mp['subfigure_file']))
    if self.transform:
      image = self.transform(image)
    return image, mp['embedding']