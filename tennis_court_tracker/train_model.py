import torch
import wandb
import hydra
import logging

from omegaconf import DictConfig

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, Rescale, ToTensor
from tennis_court_tracker.models.model import TrackNet


@hydra.main(config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    court_dataset = TennisCourtDataset(
        "data/processed/keypoints.csv", 
        "data/processed/images",
        "data/processed/labels",
        transform = transforms.Compose([
            Rescale((360, 640)),
            ToTensor(device)
        ])
    )

    train_dataset, validation_dataset = random_split(court_dataset, lengths=(0.8, 0.2))

    train_dataloader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=0)


    model = TrackNet(in_features = 3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr = config.hyperparameters.learning_rate)

    wandb.init(
        project = "tennis-court-tracking",
        config = {
            "learning_rate": config.hyperparameters.learning_rate,
            "architecture": "CNN",
            "dataset": "Custom-1",
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode # disable for testing
    )

    for epoch in range(config.hyperparameters.epochs):
        print(f"\n---- Epoch {epoch+1}/{config.hyperparameters.epochs} ----")

        total_loss = 0.0
        validation_loss = 0.0

        # Train
        model.train()
        for batch_num, batch in enumerate(train_dataloader):
            x = batch['image']
            y = batch['label']

            y_pred = model(x) # shape: [batch_size, 256, 360, 640]

            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_num % 10 == 0):
                print(f"{batch_num}/{len(train_dataloader)} | loss: {loss:.2f}")
                wandb.log({"loss": loss})
    

        # Validate
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(validation_dataloader):
                x = batch['image']
                y = batch['label']

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss.item()

                # Log a sample image
                id = 0
                if batch_num == id:
                    im_true = y[id]
                    im_pred = y_pred[id].argmax(dim=0)
                    im_stack = torch.vstack([im_true, im_pred]).cpu().float()
                    wandb.log({"sample_image": wandb.Image(im_stack, caption="Top: Input, Bottom: Output")})

        wandb.log({"train_loss": total_loss, "validation_loss" : validation_loss})
        print(f"Train loss: {total_loss:.2f},   | Val loss: {validation_loss:.2f}")
        torch.save(model.state_dict(), "models/model16.pt")

    wandb.finish()
    torch.save(model.state_dict(), "models/final_model16.pt")

if __name__ == "__main__":
    train()

