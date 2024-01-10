import torch
import wandb

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, Rescale, ToTensor
from tennis_court_tracker.models.model import TrackNet


epochs = 5
lr = 1.0
batch_size = 2
device = 'mps'

court_dataset = TennisCourtDataset(
    "data/processed/keypoints.csv", 
    "data/processed/images",
    "data/processed/labels",
    transform = transforms.Compose([
        Rescale((360, 640)),
        ToTensor(device)
    ])
)

train_dataset, test_dataset = random_split(court_dataset, lengths=(0.8,.2))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


model = TrackNet(3, 1).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr = lr)

# wandb.init(
#     project = "tennis-court-tracking",
#     config = {
#         "learning_rate": lr,
#         "architecture": "CNN",
#         "dataset": "Custom-1",
#         "epochs": epochs,
#         "batch_size" : batch_size,
#     }
# )

for epoch in range(epochs):
    print(f"\n---- Epoch {epoch+1}/{epochs} ----")

    total_loss = 0.0
    validation_loss = 0.0

    # Train
    model.train()
    for batch_num, batch in enumerate(train_dataloader):
        x = batch['image']
        y = batch['keypoints']

        y_pred = model(x) # shape: [batch_size, 1, 360, 640]

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if (batch_num % 1 == 0):
            print(f"{batch_num}/{len(train_dataloader)} | loss: {loss:.2f}")
            # wandb.log({"loss": loss})
 
    # Test
    model.eval()
    with torch.no_grad():

        for batch_num, batch in enumerate(test_dataloader):
            x = batch['image']
            y = batch['keypoints']

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            validation_loss += loss.item()

    # wandb.log({"train_loss": total_loss, "test_loss" : validation_loss})
    print(f"Train loss: {total_loss:.2f},   | Val loss: {validation_loss:.2f}")

wandb.finish()



