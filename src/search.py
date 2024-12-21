# from clip_model import model,processor
import faiss
from PIL import Image
import os
import json
from transformers import CLIPModel

d = 512
index = faiss.IndexFlatL2(d)  # 使用 L2 距离

# 保存为 JSON 文件
with open('id2filename.json', 'r') as json_file:
  id2filename = json.load(json_file)
index = faiss.read_index("image.faiss")


def text_search(text, k=1):
  inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
  text_features = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
  text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
  text_features = text_features.detach().numpy()
  D, I = index.search(text_features, k)  # 实际的查询

  filenames = [[id2filename[str(j)] for j in i] for i in I]

  return text, D, filenames


def image_search(img_path, k=1):
  image = Image.open(img_path)
  inputs = processor(images=image, return_tensors="pt")
  image_features = model.get_image_features(**inputs)
  image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

  image_features = image_features.detach().numpy()
  D, I = index.search(image_features, k)  # 实际的查询

  filenames = [[id2filename[str(j)] for j in i] for i in I]

  return img_path, D, filenames


if __name__ == "__main__":
  text = ["雪山", "熊猫", "长城", "苹果"]
  text, D, filenames = text_search(text)
  print(text, D, filenames)

  # img_path = "image/apple2.jpeg"
  # img_path,D,filenames = image_search(img_path,k=2)
  # print(img_path,D,filenames)