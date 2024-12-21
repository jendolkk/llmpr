import json
import os
import sys
import torchvision
import torch
from PIL.Image import Image
from sympy.printing.numpy import const
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
from DeepPatent2 import DP2Data
from model import AModel, VisualBackbone
import constant

trans = v2.Compose([
  v2.RandomVerticalFlip(),
  v2.RandomHorizontalFlip(),
  v2.RandomCrop(256),
  v2.RandomErasing(),
  v2.ToTensor(),
  # gridmask
])
train_data = DP2Data(constant.img_dir, constant.json_file, transform=trans)
data_loader = DataLoader(train_data, batch_size=16, shuffle=True)


device = 'cuda'
a = VisualBackbone()
model = AModel(a).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for ep in range(constant.epoch):
  for i, data in enumerate(data_loader):
    _images, _text_embeddings = data
    images = _images.to(device)
    text_embeddings = _text_embeddings.to(device)
    ls = model(images, text_embeddings)
    print(f'loss: {ls.item()}')
    optimizer.zero_grad()
    ls.backward()
    optimizer.step()

torch.save(model.state_dict(), './model.pth')