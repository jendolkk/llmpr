import random
import faiss
from PIL import Image
import os
import numpy as np
import json
import constant

dim = constant.embedding_dim
index = faiss.read_index(constant.faiss_file)

with open(constant.id2filename, 'r') as f:
  global id2filename
  id2filename = json.load(f)
with open(constant.test_json, 'r') as f:
  global test_json
  test_json = json.load(f)

def image_search(img_idx, k=5):
  img_feature = index.reconstruct(img_idx)
  img_feature = img_feature.reshape(1, -1).astype('float32')
  _, I = index.search(img_feature, k)
  return I

def calculate_map(query_num, k=5):
  average_precisions = []
  queries = np.random.randint(0, len(test_json), size=query_num)

  for img_idx in queries:
    title_id = test_json[img_idx]['title_id']
    I = image_search(img_idx, k)

    hits, precision_at_i = 0, []
    for i in I[0]:
      if test_json[i]['title_id'] == title_id:
        hits += 1
        precision_at_i.append(hits / (i + 1))

    average_precisions.append(np.mean(precision_at_i) if precision_at_i else 0)

  return np.mean(average_precisions)

def print_info(idx):
  title_id = test_json[idx]['title_id']
  locarno = test_json[idx]['classification_locarno']

  print("title_id && locarno:")
  print(title_id, locarno)
  s1 = 0
  s2 = 0
  for d in test_json:
    if d['title_id'] == title_id:
      s1 += 1
    if d['classification_locarno'] == locarno:
      s2 += 1
  print('sum')
  print(s1, s2)
  print()


# idx = random.randint(0, len(test_json))
# I = image_search(idx, k=10)
# print_info(idx)
# for i in I[0]:
#     print(i, test_json[i]['title_id'], test_json[i]['classification_locarno'])
print(calculate_map(100, 5))
print(calculate_map(100, 10))
