import json
import os
from transformers import AutoTokenizer, CLIPModel
import numpy as np
import constant
# data = json.load(constant.json_file) # list[dict]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

datas = []
for subdir in os.walk(constant.json_dir):
  for j in subdir[2]:
    print(j)
    with open(os.path.join(constant.json_dir, j), 'r', encoding='utf-8') as f:
      datas.append(json.load(f))
# with open(constant.json_, 'r', encoding='utf-8') as f:
#   data = json.load(f)

new_data = []
embeddings = []
offset = 0
for data in datas:
  for i, mp in enumerate(data):
    print(i)
    inputs = tokenizer([mp['caption'][:200]], padding=True, return_tensors='pt')
    mp['embedding_id'] = offset + i
    new_data.append(mp)
    embeddings.append(model.get_text_features(**inputs).squeeze(0).detach().numpy())
  offset += len(data)

with open('../final.json', 'w') as f:
  f.write(json.dumps(new_data))
np.save('../data/text_embeddings.npy', embeddings)