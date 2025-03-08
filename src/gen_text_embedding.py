import torch
# import clip
import json
from PIL import Image

device = "cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

data = json.load("D:\\wyh\\2016\\") # list[dict]

new_data = []
with torch.no_grad():
  for mp in data:
    name = mp['subfigure_file']
    text = clip.tokenize([]).to(device)
    mp['embedding'] = model.encode_text(text)
    new_data.append(mp)

with open('./final.json') as f:
  f.write(json.dumps(new_data, indent=2))