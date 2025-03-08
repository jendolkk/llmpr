import json
import os
import sys

import numpy as np
import constant
from transformers import BertTokenizer, BertModel

# data = json.load(constant.json_file) # list[dict]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

datas = []

for subdir in os.walk(constant.json_dir):
  for j in subdir[2]:
    with open(os.path.join(constant.json_dir, j), 'r', encoding='utf-8') as f:
      datas.append(json.load(f))

with open(constant.source_json_file, 'r', encoding='utf-8') as f:
  global mp2
  temp = json.load(f)
  mp2 = {mp["patentID"]: mp for mp in temp}

new_data = []
embeddings = []
offset = 0

for data in datas:
  for i, mp in enumerate(data):

    id = mp["patentID"]
    if id not in mp2:
      continue
    print(i)
    input = tokenizer([mp['caption'][:400]], return_tensors='pt')
    mp['embedding_id'] = offset
    mp["object_title"] = mp2[id]["object_title"]
    mp["classification_locarno"] = mp2[id]["classification_locarno"]
    new_data.append(mp)
    # embeddings.append(model.get_text_features(**inputs).squeeze(0).detach().numpy())
    ebd = model(**input).last_hidden_state[:,0,:]
    embeddings.append(ebd.squeeze(0).detach().numpy())
    offset += 1

with open(constant.json_file, 'w') as f:
  json.dump(new_data, f)
np.save('../data/text_embeddings.npy', embeddings)