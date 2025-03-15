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

def calculate_recall(k=5):
  total_relevant, total_retrieved_relevant = 0, 0
  for query in test_json:
    img_idx = query['image_id']
    relevant_images = set(query['relevant_images'])
    retrieved_images = set(image_search(img_idx, k))
    total_relevant += len(relevant_images)
    total_retrieved_relevant += len(retrieved_images & relevant_images)
  return total_retrieved_relevant / total_relevant if total_relevant > 0 else 0


def calculate_map(k=5):
  average_precisions = []
  for query in test_json:
    img_idx = query['image_id']
    relevant_images = set(query['relevant_images'])
    retrieved_images = image_search(img_idx, k)

    hits, precision_at_i = 0, []
    for i, img in enumerate(retrieved_images):
      if img in relevant_images:
        hits += 1
        precision_at_i.append(hits / (i + 1))

    average_precisions.append(np.mean(precision_at_i) if precision_at_i else 0)

  return np.mean(average_precisions)

if __name__ == "__main__":
  img_idx = random.randint(len(test_json))
  filenames = image_search(img_idx, k=5)