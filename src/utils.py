import json
import torch
from tqdm import tqdm
from src.model import AModel, VisualBackbone



def load_model(name, path):
  model = AModel(VisualBackbone(name)).to('cuda')
  model.load_state_dict(torch.load(path))
  model.eval()
  return model

def load_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)

def get_embeddings(data_loader, model, trans, device='cuda'):
  res1 = []
  res2 = []
  loop = tqdm(data_loader, desc=f'runnning', total=len(data_loader), leave=True)
  for batch_idx, (images, text_embeddings, titles, heads) in enumerate(loop):
    images, titles = (
      images.to(device),
      titles.to(device),
    )
    img_features = model.visual_encoder(trans(images).to(device)).cpu()
    res1.append(img_features)
    res2.append(titles)
  return torch.vstack(res1), torch.vstack(res2)