import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.cuda import temperature
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision.models import resnet50, ResNet50_Weights


class AModel(nn.Module):
  def __init__(self, visual_backbone, temperature=0.07):
    super().__init__()
    # visual_backbone = visual_backbone
    self.visual_encoder = visual_backbone
    self.temperature = nn.Parameter(torch.tensor(temperature))
    self.s_hats = nn.Parameter(torch.zeros(3))

  def forward(self, images, texts_embeddings):
    visual_embeddings = self.visual_encoder(images)
    # 计算损失函数
    clip_loss = self.calculate_clip_loss(visual_embeddings, texts_embeddings)
    cls_loss, cls_labels = self.calculate_cls_loss(visual_embeddings, texts_embeddings)
    cat_loss, cat_labels = self.calculate_cat_loss(visual_embeddings, texts_embeddings)

    loss3 = torch.tensor([-clip_loss, cls_loss, cat_loss])
    total_loss = (loss3 / self.s_hats.exp()).sum() + self.s_hats.sum()
    return total_loss

  def calculate_clip_loss(self, visual_embeddings, text_embeddings):
    logits = (text_embeddings @ image_embeddings.T) / self.temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
      (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
    )
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
    return loss.mean()

    # similarities = F.cosine_similarity(visual_embeddings, text_embeddings, dim=1)
    # logits_per_image = torch.exp(similarities) / torch.sum(torch.exp(similarities), dim=1, keepdim=True)
    # loss = -torch.log(logits_per_image.gather(1, text_embeddings.argmax(1).unsqueeze(1)).squeeze())
    # return loss.mean()

  def calculate_cls_loss(self, visual_features, text_features, cls_labels):
    # 假设cls_labels是类别标签，用于计算基于类别的损失
    class_logits = torch.matmul(visual_features, text_features.T) / self.temperature
    class_loss = F.cross_entropy(class_logits, cls_labels)
    return class_loss, cls_labels

  def calculate_cat_loss(self, visual_features, text_features, cat_labels):
    # 假设cat_labels是类别标签，用于计算基于头部/尾部类别的损失
    category_logits = torch.matmul(visual_features, text_features.T) / self.temperature
    category_loss = F.cross_entropy(category_logits, cat_labels)
    return category_loss, cat_labels

class VisualBackbone(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.resnet.fc = nn.Linear(2048, 512)

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