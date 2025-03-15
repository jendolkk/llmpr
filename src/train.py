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
  # v2.Resize((256,256)),
  v2.Resize((224,224)),
  v2.ToTensor(),
  # gridmask
])
train_data = DP2Data(constant.img_dir, constant.train_json, transform=trans)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_data = DP2Data(constant.img_dir, constant.val_json, transform=trans)
val_loader = DataLoader(val_data, batch_size=256, shuffle=True)

writer = SummaryWriter(log_dir="./logs3")

device = 'cuda'
a = VisualBackbone('vit')
model = AModel(a).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

for ep in range(constant.epoch):
  loop = tqdm(train_loader, desc=f'Epoch {ep + 1}/{constant.epoch}', total=len(train_loader), leave=True)
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
    # writer.add_scalar("Batch Loss", loss.item(), ep * len(train_loader) + batch_idx)

  avg_loss = epoch_loss / len(train_loader)
  writer.add_scalar("Epoch Loss", avg_loss, ep)

  if ep > 20:
    torch.save(model.state_dict(), f'../models2/model{ep}.pth')
    torch.save(optimizer.state_dict(), f'../models2/opt{ep}.pth')
  with torch.no_grad():
    loop = tqdm(val_loader, desc=f'Epoch_val {ep + 1}/{constant.epoch}', total=len(val_loader), leave=False)
    val_loss = 0
    for batch_idx, (images, text_embeddings, titles, heads) in enumerate(loop):
      images, text_embeddings, titles, heads = (
        images.to(device),
        text_embeddings.to(device),
        titles.to(device),
        heads.to(device),
      )

      loss = model(images, text_embeddings, titles, heads)

      val_loss += loss.item()
      # writer.add_scalar("Batch Loss", loss.item(), ep * len(train_loader) + batch_idx)

    avg_loss = val_loss / len(val_loader)
    writer.add_scalar("Epoch Val Loss", avg_loss, ep)

writer.close()
# torch.save(model.state_dict(), './model.pth')