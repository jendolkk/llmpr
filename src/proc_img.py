import json
import os
import torch
from PIL import Image
import numpy as np
from modelb import AModel, VisualBackbone
import faiss
import constant

d = constant.embedding_dim
index = faiss.IndexFlatL2(d)
id2filename = []
img_names = []
imgf_np = []

model = AModel(VisualBackbone())
model.load_state_dict(torch.load('./model.pth'))
model.eval()

transf = torch.transform.v2.ToTensor()
with torch.no_grad():
  with open(constant.json_file, 'r') as f:
    dicts = json.load(f)
    for mp in dicts:
      name = mp['subfigure_file']
      image = Image.open(os.path.join(constant.img_dir, name))
      img_feature = model.visual_encoder(transf(image).unsqueeze(0)).squeeze(0).numpy()
      index.add(img_feature)
      imgf_np.append(img_feature)
      img_names.append(name)

np.save(constant.imgf_np, imgf_np)
faiss.write_index(index, constant.faiss_file)