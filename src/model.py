import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights, vit_b_32, ViT_B_32_Weights
import constant


class AModel(nn.Module):
  def __init__(self, visual_backbone):
    super().__init__()
    self.visual_encoder = visual_backbone
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.s_hats = nn.Parameter(torch.zeros(3))

  def forward(self, images, texts_embeddings, cls, cat):
    visual_embeddings = self.visual_encoder(images)
    visual_embeddings = visual_embeddings / visual_embeddings.norm(dim=1, keepdim=True)
    texts_embeddings = texts_embeddings / texts_embeddings.norm(dim=1, keepdim=True)

    logit_scale = self.logit_scale.exp()
    logits_per_text = (texts_embeddings @ visual_embeddings.T) * logit_scale
    logits_per_img = logits_per_text.T

    loss1, loss2 = self.calculate_loss(logits_per_text, logits_per_img, torch.arange(len(images), device=images.device))
    clip_loss = (loss1 + loss2) / 2.0
    cls_target = (cls.unsqueeze(0) == cls.unsqueeze(1)).float()
    cat_target = (cat.unsqueeze(0) == cat.unsqueeze(1)).float()
    loss3, loss4 = self.calculate_loss(logits_per_text, logits_per_img, cls_target)
    cls_loss = (loss3 + loss4) / 2.0
    loss5, loss6= self.calculate_loss(logits_per_text, logits_per_img, cat_target)
    cat_loss = (loss4 + loss6) / 2.0
    loss_all = torch.stack([clip_loss, cls_loss, cat_loss])

    total_loss = (loss_all * (-self.s_hats.to(loss_all.device)).exp()).sum() + self.s_hats.to(loss_all.device).sum()
    return total_loss

  def calculate_loss(self, logits_per_text, logits_per_img, labels):
    caption_loss = self.contrastive_loss(logits_per_text, labels)
    image_loss = self.contrastive_loss(logits_per_img, labels)

    return caption_loss, image_loss

  def contrastive_loss(self, logits: torch.Tensor, target) -> torch.Tensor:
    return F.cross_entropy(logits, target)

class VisualBackbone(nn.Module):
  def __init__(self, base: str, pretrained=True):
    super().__init__()
    if base == "resnet":
      self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
      self.encoder.fc = nn.Linear(2048, constant.embedding_dim)
    elif base == "vit":
      self.encoder = vit_b_32(weights=ViT_B_32_Weights.DEFAULT if pretrained else None)
      self.encoder.heads = nn.Identity()

  def forward(self, x):
    return self.encoder(x)