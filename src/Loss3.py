import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss3(nn.Module):
  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature
    # 可学习的同方差不确定性参数
    self.log_vars = nn.Parameter(torch.zeros(3))

  def l_clip(self, image_features, text_features):
    batch_size = image_features.shape[0]
    similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1) / self.temperature
    positive_scores = similarity_matrix.diag().view(-1, 1)
    negative_scores = similarity_matrix + torch.diag(torch.ones(batch_size) * -10000).to(similarity_matrix.device)
    exp_negative_scores = torch.exp(negative_scores)
    L_clip = -torch.log(positive_scores / exp_negative_scores.sum(dim=1, keepdim=True))


  # im_f [batch_size, dim], tx_f [batch, dim]
  def forward(self, image_features, text_features):

    # 计算类别内损失（L_cls）和类别间损失（L_cat）
    L_cls = 0
    L_cat = 0
    for i in range(batch_size):
      # 找到与当前图像属于同一类别的文本特征索引
      cls_positive_indices = (text_features[i] == text_features).all(dim=-1)
      cls_positive_scores = similarity_matrix[i][cls_positive_indices]
      cls_negative_scores = similarity_matrix[i][~cls_positive_indices]
      exp_cls_negative_scores = torch.exp(cls_negative_scores)
      L_cls -= torch.log(cls_positive_scores / (exp_cls_negative_scores.sum() + cls_positive_scores.sum()))

      # 找到与当前图像属于同一类别的图像特征索引（这里假设类别信息可以从图像特征中获取，实际情况可能需要根据数据集的具体情况调整）
      cat_positive_indices = (image_features[i] == image_features).all(dim=-1)
      cat_positive_scores = similarity_matrix[i][cat_positive_indices]
      cat_negative_scores = similarity_matrix[i][~cat_positive_indices]
      exp_cat_negative_scores = torch.exp(cat_negative_scores)
      L_cat -= torch.log(cat_positive_scores / (exp_cat_negative_scores.sum() + cat_positive_scores.sum()))

    # 根据可学习的同方差不确定性参数调整损失
    L_clip = L_clip * torch.exp(-self.log_vars[0]) + self.log_vars[0]
    L_cls = L_cls * torch.exp(-self.log_vars[1]) + self.log_vars[1]
    L_cat = L_cat * torch.exp(-self.log_vars[2]) + self.log_vars[2]

    # 总损失
    total_loss = L_clip + L_cls + L_cat

    return total_loss