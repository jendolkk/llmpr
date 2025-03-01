import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.special import logit
from torchvision.models import resnet50, ResNet50_Weights
import constant


class AModel(nn.Module):
  def __init__(self, visual_backbone):
    super().__init__()
    # visual_backbone = visual_backbone
    self.visual_encoder = visual_backbone
    # self.temperature = nn.Parameter(torch.tensor(temperature))
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.s_hats = nn.Parameter(torch.zeros(3))

  def forward(self, images, texts_embeddings, cls, cat):
    visual_embeddings = self.visual_encoder(images)
    loss1, loss2 = self.calculate_loss(visual_embeddings, texts_embeddings, torch.arange(len(images)))
    clip_loss = (loss1 + loss2) / 2.0
    loss3, loss4 = self.calculate_loss(visual_embeddings, texts_embeddings, cls)
    cls_loss = (loss3 + loss4) / 2.0
    loss5, loss6= self.calculate_loss(visual_embeddings, texts_embeddings, cat)
    cat_loss = (loss4 + loss6) / 2.0
    loss3 = torch.tensor([clip_loss, cls_loss, cat_loss])
    total_loss = (loss3 * self.s_hats.exp()).sum() + self.s_hats.sum()
    return total_loss

  # def cross_entropy(self, preds, targets, reduction='none'):
  #   log_softmax = nn.LogSoftmax(dim=-1)
  #   loss = (-targets * log_softmax(preds)).sum(1)
  #   if reduction == "none":
  #     return loss
  #   elif reduction == "mean":
  #     return loss.mean()

  def calculate_loss(self, visual_embeddings, text_embeddings, labels):
    visual_embeddings = visual_embeddings / visual_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    logits_per_text = (text_embeddings @ visual_embeddings.T) * self.logit_scale
    logits_per_img = logits_per_text.T

    caption_loss = self.contrastive_loss(logits_per_text, labels)
    image_loss = self.contrastive_loss(logits_per_img, labels)

    return caption_loss, image_loss

  def contrastive_loss(self, logits: torch.Tensor, labels) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class VisualBackbone(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.resnet.fc = nn.Linear(2048, constant.embedding_dim)

  def forward(self, x):
    return self.resnet(x)


# class ProjectionHead(nn.Module):
#   def __init__(
#       self,
#       embedding_dim,
#       projection_dim=512,
#       dropout=0.1
#   ):
#     super().__init__()
#     self.projection = nn.Linear(embedding_dim, projection_dim)
#     self.gelu = nn.GELU()
#     self.fc = nn.Linear(projection_dim, projection_dim)
#     self.dropout = nn.Dropout(dropout)
#     self.layer_norm = nn.LayerNorm(projection_dim)
#
#   def forward(self, x):
#     projected = self.projection(x)
#     x = self.gelu(projected)
#     x = self.fc(x)
#     x = self.dropout(x)
#     x = x + projected
#     x = self.layer_norm(x)
#     return x