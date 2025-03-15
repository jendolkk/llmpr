import json
import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from modelb import AModel, VisualBackbone
import faiss
import constant

d = constant.embedding_dim
index = faiss.IndexFlatL2(d)
id2filename = {}
filename2id = {}
# imgf_np = []

model = AModel(VisualBackbone('vit'))
model.load_state_dict(torch.load('../models/model19.pth'))
model.eval()

trans = v2.Compose([
  v2.Resize((224, 224)),
  v2.ToTensor(),
])

with torch.no_grad():
  with open(constant.test_json, 'r') as f:
    dicts = json.load(f)
    for i, mp in enumerate(dicts):
      name = mp['subfigure_file']
      id2filename[i] = name
      filename2id[name] = i

      image = Image.open(os.path.join(constant.img_dir, name))
      img_feature = model.visual_encoder(trans(image).unsqueeze(0)).squeeze(0).numpy()
      index.add(img_feature)
      # imgf_np.append(img_feature)

# np.save(constant.imgf_np, imgf_np)
faiss.write_index(index, constant.faiss_file)