# from clip_model import model,processor
import random

import faiss
from PIL import Image
import os
import numpy as np
import json

from matplotlib import image as mpimg
from transformers import CLIPModel
import constant
from src.proc_img import id2filename
import matplotlib as plt
import constant

dim = 512
# index = faiss.IndexFlatL2(dim)
index = faiss.read_index("image.faiss")
id2filename = None
imgf_np = np.load(constant.imgf_np)
with open('id2filename.json', 'r') as json_file:
  id2filename = json.load(json_file)

def image_search(img_idx, k=5):
  assert img_idx < len(id2filename)
  image_features = imgf_np[img_idx]
  dists, idxs = index.search(image_features, k)
  filenames = [[[os.path.join(constant.img_dir, str(j))] for j in i] for i in idxs]
  return filenames

if __name__ == "__main__":
  img_idx = random.randint(len(id2filename))
  filenames = image_search(img_idx, k=5)
  fig, axes = plt.subplots(1, 5, figsize=(15, 5))
  for ax, image_path in zip(axes, filenames):
      img = mpimg.imread(image_path)
      ax.imshow(img)
      ax.axis('off')
  plt.show()