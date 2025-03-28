import json
import os
from transformers import AutoTokenizer, CLIPModel
import numpy as np
import constant

src = constant.source_json_file
dest_train = constant.clip_task_train_json
dest_test = constant.clip_task_test_json
dicts = []
mp = {}
with open(src, 'r', encoding='utf-8') as f:
  u = json.load(f)
  for i, m in enumerate(u):
    idx = -1
    if m['object_title'] in mp:
      idx = m['object_title']
    else:
      idx = len(mp)
      mp['object_title'] = idx
    assert idx != -1

    m['title_id'] = idx
    dicts.append(m)

    if i == 120000:
      break

with open(dest_train, 'w', encoding='utf-8') as f:
  json.dump(dicts[:100000], f, indent=2)
with open(dest_test, 'w', encoding='utf-8') as f:
  json.dump(dicts[100000:], f, indent=2)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14").to('cuda')

texts = [''] * len(mp)
for k, v in mp:
  texts[v] = f"a photo of a {k}"

B = 200
embeddings = []
for i in range(len(mp) // B):
  inputs = tokenizer(texts[B*i:B*(i+1)], padding=True, return_tensors='pt')
  text_features = model.get_text_features(**inputs).cpu().detach().numpy()
  embeddings.append(text_features)

res = np.vstack(embeddings)
np.save('../data/text_embeddings.npy', res)
print(res.shape)
