import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from tqdm import tqdm

from DeepPatent2 import DP2Data
from model import AModel, VisualBackbone
import constant

trans = v2.Compose([
  # v2.RandomVerticalFlip(),
  # v2.RandomHorizontalFlip(),
  # v2.RandomCrop(256),
  # v2.RandomErasing(),
  v2.Resize((256,256)),
  v2.ToTensor(),
  # gridmask
])
train_data = DP2Data(constant.img_dir, constant.ok_file, transform=trans)
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

writer = SummaryWriter(log_dir="./logs")

device = 'cuda'
a = VisualBackbone()
model = AModel(a).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

model.train()

for ep in range(constant.epoch):
  loop = tqdm(data_loader, desc=f'Epoch {ep + 1}/{constant.epoch}', total=len(data_loader), leave=True)
  epoch_loss = 0
  for batch_idx, (images, text_embeddings, titles, heads) in enumerate(loop):
    images, text_embeddings, titles, heads = (
      images.to(device),
      text_embeddings.to(device),
      titles.to(device),
      heads.to(device),
    )

    loss = model(images, text_embeddings, titles, heads)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

    writer.add_scalar("Batch Loss", loss.item(), ep * len(data_loader) + batch_idx)

  avg_loss = epoch_loss / len(data_loader)
  writer.add_scalar("Epoch Loss", avg_loss, ep)

writer.close()
torch.save(model.state_dict(), './model.pth')