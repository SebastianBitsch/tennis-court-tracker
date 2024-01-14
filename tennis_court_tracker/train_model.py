import torch
import wandb
import hydra
import logging

from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, RandomCrop, TransformWrapper
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
            # TransformWrapper(transforms.Resize((config.data.image_height, config.data.image_width), antialias=True))
            RandomCrop((config.data.image_height, config.data.image_width)),
        ])
    )

    train_dataset, validation_dataset = random_split(court_dataset, lengths = (config.data.pct_train_split, 1.0 - config.data.pct_train_split))
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=0)

    model = TrackNet(in_features = config.data.n_in_features).to(device)
    # Load model weights
    if config.hyperparameters.continue_training_from_weights:
        logger.info(f"Loading model state dict at: {config.hyperparameters.path_to_weights}")
        model.load_state_dict(torch.load(config.hyperparameters.path_to_weights))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr = config.hyperparameters.learning_rate)

    wandb.init(
        project = config.wandb.project_name,
        config = {
            "architecture": model.name,
            "dataset": "Custom-1",
            "learning_rate": config.hyperparameters.learning_rate,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )


    for epoch in range(config.hyperparameters.epochs):
        logger.info(f"**** Epoch {epoch+1}/{config.hyperparameters.epochs} ****")

        training_loss = 0.0
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
            training_loss += loss.item()

            if (batch_num % config.wandb.train_log_interval == 0):
                logger.info(f"{batch_num + 1}/{len(train_dataloader)} | loss: {loss:.3f}")
                wandb.log({"training_loss": training_loss / (batch_num + 1)})

        # Validate
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(validation_dataloader):
                x = batch['image']
                y = batch['heatmap']

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss.item()

                if (batch_num % config.wandb.validation_log_interval == 0):
                    logger.info(f"{batch_num + 1}/{len(validation_dataloader)} | val loss: {loss.item():.3f}")
                    
                    id = torch.randint(low=0, high=x.shape[0], size=(1,)).item()
                    im = x[id]
                    im_true = y[id].float().repeat(3,1,1)
                    im_pred = y_pred[id].argmax(dim=0).float().repeat(3,1,1)

                    grid = make_grid([im, im_true, im_pred])
                    wandb.log({
                        "validation_loss": validation_loss / (batch_num + 1),
                        "sample_image": wandb.Image(grid, caption="Left: Input | Middle: Labels | Right: Predicted")
                    })

        torch.save(model.state_dict(), "models/model.pt")

    wandb.finish()
    torch.save(model.state_dict(), "models/final_model.pt")



if __name__ == "__main__":
    train()

