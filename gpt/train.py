import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from gpt.A3CDataset import A3CDataset
from gpt.GPT import GPT
from gpt.Tokenizer import Tokenizer
from gpt.config import CFG

# set seed of all polssible random number generators
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# enable CUDA_LAUNCH_BLOCKING=1 to debug cuda
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


## LOAD DATA
def read_data_frame():
    df = pd.read_csv("data/all_data.csv")
    episodes = df.groupby("Episode")
    episode_data = {}
    for episode, data in episodes:
        episode_data[episode] = [
            [row["X"], row["Y"], row["Action"], row["Reward"]]
            for index, row in data.iterrows()
        ]
    return episode_data


episode_data = [episode for episode in read_data_frame().values()]

## SPLIT DATA
np.random.shuffle(episode_data)
train_data = episode_data[: int(len(episode_data) * 0.8)]
test_data = episode_data[int(len(episode_data) * 0.8) :]


## LOADERS
tokenizer = Tokenizer()
train_data_loader = DataLoader(
    A3CDataset(train_data, tokenizer=tokenizer),
    batch_size=CFG.batch_size,
    shuffle=False,
)
test_data_loader = DataLoader(
    A3CDataset(test_data, tokenizer=tokenizer), batch_size=CFG.batch_size, shuffle=False
)


model = GPT() 

m = model.to(CFG.device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")


optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)


cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, CFG.epoch * len(train_data_loader), eta_min=3e-7 
)

writer = SummaryWriter()
# training loop
for epoch in range(CFG.epoch):
    model.train()
    writer.add_scalar("Epoch", epoch, epoch)
    for batch, (X, y) in tqdm(
        enumerate(train_data_loader),
        unit="batch",
        total=len(train_data_loader),
        desc=f"Epoch {epoch}",
    ):
        X = X.to(CFG.device)
        y = y.to(CFG.device)
        optimizer.zero_grad()
        logits, loss = model(X, y)
        writer.add_scalar("Loss/train", loss.item(), epoch * len(train_data_loader) + batch)
        writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]["lr"],
            epoch * len(train_data_loader) + batch,
        ) 
        loss.backward() 
        optimizer.step()
        cosine_schedule.step()
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch, (X, y) in tqdm(
            enumerate(test_data_loader),
            unit="batch",
            total=len(test_data_loader),
            desc=f"Epoch {epoch}",
        ):
            X = X.to(CFG.device)
            y = y.to(CFG.device)
            logits, loss = model(X, y)
            total_correct += (logits.argmax(1) == y).sum().item()
            writer.add_scalar("Loss/val", loss, epoch * len(test_data_loader) + batch) 
        writer.add_scalar(
            "Accuracy/val", total_correct / (len(test_data_loader) * CFG.batch_size), epoch 
        )
        # add accuracy to tqdm
        tqdm.write(f"Epoch {epoch}, Total correct: {total_correct}, Length: {len(test_data_loader)}, Accuracy: {total_correct / (len(test_data_loader) * CFG.batch_size)}")

# save the model
model_path = "models/gpt_model.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)


writer.close()
