import json
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from model import AModel, VisualBackbone
import faiss
import constant
from src.DeepPatent2 import DP2Data

d = constant.embedding_dim
index = faiss.IndexFlatIP(d)
id2filename = {}
filename2id = {}

trans = v2.Compose([
  v2.Resize((224, 224)),
  v2.ToTensor(),
])

# test_data = DP2Data(constant.img_dir, constant.test_json, transform=trans)
# test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

model = AModel(VisualBackbone('vit')).to('cuda')
model.load_state_dict(torch.load('../models2/model48.pth'))
model.eval()


with torch.no_grad():
  with open(constant.test_json, 'r') as f:
    dicts = json.load(f)
    for i, mp in enumerate(dicts):
      print(i)
      name = mp['subfigure_file']
      id2filename[i] = name
      filename2id[name] = i

      image = Image.open(os.path.join(constant.img_dir, name))
      img_feature = model.visual_encoder(trans(image).to('cuda').unsqueeze(0)).squeeze(0).cpu().numpy()
      img_feature = img_feature.reshape(1, -1)
      faiss.normalize_L2(img_feature)
      index.add(img_feature)
      # imgf_np.append(img_feature)

# np.save(constant.imgf_np, imgf_np)
faiss.write_index(index, constant.faiss_file)
with open(constant.id2filename, 'w', encoding='utf-8') as f:
  json.dump(id2filename, f, indent=2)
with open(constant.filename2id, 'w', encoding='utf-8') as f:
  json.dump(filename2id, f, indent=2)
