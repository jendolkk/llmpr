import json
import os
import random

from transformers import AutoTokenizer, CLIPModel
import numpy as np
import constant

src = constant.source_json_file
dest_train = constant.clip_task_train_json
dest_val = constant.clip_task_val_json

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14").to('cuda')

texts = []
sz = 0
with open(src, 'r', encoding='utf-8') as f1, \
     open(dest_train, 'r', encoding='utf-8') as f2, \
     open(dest_val, 'r', encoding='utf-8') as f3:
  u = json.load(f1)
  for d in u:
    sz = max(sz, d["title_id"])
  texts = [''] * (sz + 1)
  for d in u:
    id = d["title_id"]
    if texts[id] == '':
      texts[id] = f"a photo of a {d['object_title'].strip().lower()}"
  random.seed(42)
  random.shuffle(u)
  idx = len(u) * 0.9
  json.dump(u[:idx], f2)
  json.dump(u[idx:], f3)


B = 1024
embeddings = []
for i in range((sz + B - 1) // B):
  inputs = tokenizer(texts[B*i:B*(i+1)], padding=True, return_tensors='pt')
  text_features = model.get_text_features(**inputs).cpu().detach().numpy()
  embeddings.append(text_features)

res = np.vstack(embeddings)
np.save('../data/text_embeddings.npy', res)
print(res.shape)
