import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.cuda import temperature
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision.models import resnet50, ResNet50_Weights

from src import constant


class AModel(nn.Module):
  def __init__(self, visual_backbone, temperature=0.07):
    super().__init__()
    # visual_backbone = visual_backbone
    self.visual_encoder = visual_backbone
    self.temperature = nn.Parameter(torch.tensor(temperature))
    self.s_hats = nn.Parameter(torch.ones(3) /2).to('cuda')

  def forward(self, images, texts_embeddings, cls, cat):
    visual_embeddings = self.visual_encoder(images)
    clip_loss = self.calculate_clip_loss(visual_embeddings, texts_embeddings).item()
    cls_loss = self.calculate_loss(visual_embeddings, texts_embeddings, cls).item()
    cat_loss = self.calculate_loss(visual_embeddings, texts_embeddings, cat).item()
    loss3 = torch.tensor([-clip_loss, cls_loss, cat_loss]).to('cuda')
    total_loss = (loss3 / self.s_hats.exp()).sum() + self.s_hats.sum()
    return total_loss

  def cross_entropy(self, preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
      return loss
    elif reduction == "mean":
      return loss.mean()

  def calculate_clip_loss(self, visual_embeddings, text_embeddings):
    logits = (text_embeddings @ visual_embeddings.T) / self.temperature
    images_similarity = visual_embeddings @ visual_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
      (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
    )
    texts_loss = self.cross_entropy(logits, targets, reduction='none')
    images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
    return loss.mean()


  def calculate_loss(self, visual_embeddings, text_embeddings, labels):
    logits_per_text = (text_embeddings @ visual_embeddings.T) / self.temperature
    logits_per_text = logits_per_text.exp()
    sim_mat = (labels == labels.T).int()
    # logits_per_img = logits_per_text.T
    logits2 = logits_per_text * sim_mat
    # logits2_T = logits2.T

    loss_per_text = logits2.sum(1) / logits_per_text.sum(1)
    loss_per_img = logits2.sum(0) / logits_per_text.sum(0)
    denominator = sim_mat.sum(0)
    return (-(loss_per_text / denominator).log() + -(loss_per_img / denominator).log()).sum()

class VisualBackbone(nn.Module):
  def __init__(self):
    super().__init__()
    self.visual_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.visual_encoder.fc = nn.Linear(2048, constant.embedding_dim)

  def forward(self, x):
    return self.visual_encoder(x)


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