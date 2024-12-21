import json
import os
import torch
from PIL import Image
from model import AModel, VisualBackbone
import faiss
import constant

d = 512
index = faiss.IndexFlatL2(d)
img_names = []

model = AModel(VisualBackbone())
model.load_state_dict(torch.load('./model.pth'))
model.eval()

transf = torch.transform.v2.ToTensor()
with torch.no_grad():
  with open(constant.json_file, 'r') as f:
    dicts = json.load(f)
    for mp in dicts:
      name = mp['subfigure_file']
      image = Image.open(os.path.join(constant.img_dir, mp['subfigure_file']))
      img_feature = model.resnet(transf(image).unsqueeze(0)).squeeze(0).numpy()
      index.add(img_feature)
      img_names.append(name)

faiss.write_index(index, constant.faiss_file)