import torch
import torchvision.transforms.v2 as v2
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader

from src import constant
from src.DeepPatent2 import DP2Data
from src.model import AModel, VisualBackbone
from src.utils import get_embeddings
from utils import load_model, load_json


model = load_model('resnet', '../model1/model20.pth')
test_json = load_json(constant.clip_task_test_json)

trans = v2.Compose([
  # v2.Resize((256,256)),
  v2.Resize((224, 224)),
  v2.ToTensor(),
])
test_data = DP2Data(constant.img_dir, constant.clip_task_test_json, transform=trans)
test_loader = DataLoader(test_data, batch_size=200, shuffle=False)

embeddings, title_ids = None, None

with torch.no_grad():
    embeddings, title_ids = get_embeddings(test_loader, model, trans)


print(title_ids[:10])

calculator = AccuracyCalculator(
  include=['mean_average_precision', 'mean_reciprocal_rank'],
  avg_of_avgs=True,
  k=10,
  device=torch.device('cuda')
)

query = []
query_labels = []
s = set()
for i in range(len(embeddings)):
  if title_ids[i] not in s:
    query.append(embeddings[i])
    query_labels.append(title_ids[i])
    s.add(title_ids[i])

query = torch.vstack(query)
query_labels = torch.vstack(query_labels)
# print(query_labels[:10])
dicts = calculator.get_accuracy(
  query, query_labels,
  embeddings, title_ids,
  # query, query_labels,
  ref_includes_query=False
)

print(dicts)