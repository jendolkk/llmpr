import torch
import torchvision.transforms.v2 as v2
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader

from src import constant
from src.DeepPatent2 import DP2Data
from src.model import AModel, VisualBackbone
from src.utils import get_embeddings
from utils import load_model, load_json


model = load_model('resnet', '../models1/model20.pth')
test_json = load_json(constant.clip_task_test_json)

trans = v2.Compose([
  # v2.Resize((256,256)),
  v2.Resize((224, 224)),
  v2.ToTensor(),
])
test_data = DP2Data(constant.img_dir, constant.clip_task_test_json, transform=trans)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

embeddings, title_ids = get_embeddings(test_loader, model, trans)

calculator = AccuracyCalculator(
  include=['mean_average_precision', 'mean_reciprocal_rank'],
  avg_of_avgs=True,
  device='cuda'
)

query = []
query_labels = []
s = set()
for i in range(len(embeddings)):
  if title_ids[i] not in s:
    query.append(embeddings[i])
    query_labels.append(title_ids[i])
    s.add(title_ids[i])

dicts = calculator.get_accuracy(
  query, query_labels,
  embeddings, title_ids,
  ref_includes_query=True
)

print(dicts)