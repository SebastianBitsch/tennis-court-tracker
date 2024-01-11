import torch
import wandb
import hydra
import logging

from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset#, Rescale, ToTensor
from tennis_court_tracker.models.model import TrackNet

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    court_dataset = TennisCourtDataset(
        annotations_file_path = to_absolute_path("data/raw/data_train.json"), 
        images_dir = to_absolute_path("data/raw/images"),
        device = device,
        transform = transforms.Compose([
            transforms.Resize((config.data.image_height, config.data.image_width), antialias=True)
            # transforms.RandomCrop((config.data.image_height, config.data.image_width), antialias=True)
        ])
    )

    train_dataset, validation_dataset = random_split(court_dataset, lengths = (config.data.pct_train_split, 1.0 - config.data.pct_train_split))
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=0)

    model = TrackNet(in_features = config.data.n_in_features).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr = config.hyperparameters.learning_rate)

    wandb.init(
        project = config.wandb.project_name,
        config = {
            "architecture": "CNN",
            "dataset": "Custom-1",
            "learning_rate": config.hyperparameters.learning_rate,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )


    for epoch in range(config.hyperparameters.epochs):
        logger.info(f"**** Epoch {epoch+1}/{config.hyperparameters.epochs} ****")

        total_loss = 0.0
        validation_loss = 0.0

        # Train
        model.train()
        for batch_num, batch in enumerate(train_dataloader):
            x = batch['image']
            y = batch['heatmap']

            y_pred = model(x) # shape: [batch_size, 256, image_height, image_width]

            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_num % 1 == 0):
                logger.info(f"{batch_num + 1}/{len(train_dataloader)} | loss: {loss:.3f}")
                wandb.log({"loss": loss})
            break

        # Validate
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(validation_dataloader):
                x = batch['image']
                y = batch['heatmap']

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss.item()
                logger.info(f"{batch_num + 1}/{len(train_dataloader)} | val loss: {loss.item():.3f}")

                # Log a sample image
                id = 0
                if batch_num == id:
                    im_true = y[id]
                    im_pred = y_pred[id].argmax(dim=0)
                    im_stack = torch.hstack([im_true, im_pred]).cpu().float()
                    wandb.log({"sample_image": wandb.Image(im_stack, caption="Top: Input, Bottom: Output")})


        wandb.log({"train_loss": total_loss, "validation_loss" : validation_loss})
        logger.info(f"Train loss: {total_loss:.3f},   | Val loss: {validation_loss:.3f}")
        torch.save(model.state_dict(), "models/model16.pt")

    wandb.finish()
    torch.save(model.state_dict(), "models/final_model16.pt")


if __name__ == "__main__":
    train()

