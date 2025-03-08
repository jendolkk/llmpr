from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import constant

class DP2Data(Dataset):
  def __init__(self, image_dir, json_file, transform=None):
    self.image_dir = image_dir
    self.image_list = os.listdir(image_dir)
    import json
    with open(json_file, 'r') as f:
      self.dicts = json.load(f)
    self.ebd = np.load(constant.embedding_file)
    self.transform = transform

  def __len__(self):
    return len(self.dicts)

  def __getitem__(self, idx):
    mp = self.dicts[idx]
    image = Image.open(os.path.join(self.image_dir, mp['subfigure_file']))
    if self.transform:
      image = self.transform(image)
    return image, self.ebd[mp['embedding_id']], mp["title_id"], mp.get('head', 0)