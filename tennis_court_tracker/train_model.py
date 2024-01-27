import torch
import wandb
import hydra
import logging
from itertools import chain

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, RandomCrop, TransformWrapper, Normalize, RandomAffine
from tennis_court_tracker.models.tracknet import TrackNet

logger = logging.getLogger(__name__)


def make_image_grid(x, y, y_pred):
    i = [im for im in x]
    
    heatmaps_true, _ = y.max(dim=1)
    t = [im.repeat(3,1,1) for im in heatmaps_true]

    heatmaps_pred, _ = y_pred.max(dim=1)
    p = [im.repeat(3,1,1) for im in heatmaps_pred]

    all_ims = list(chain.from_iterable(zip(i,t,p)))
    grid = make_grid(all_ims, nrow=3) # (3, n*W, n*H)
    return grid


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    court_dataset = TennisCourtDataset(
        annotations_file_path = to_absolute_path(config.data.annotations_file_path), 
        images_dir = to_absolute_path(config.data.images_dir_path),
        device = device,
        transform = transforms.Compose([
            RandomAffine(im_size=(config.data.image_height, config.data.image_width), degrees = (-15, 15), translate = (0.2, 0.2), scale=(0.5, 1.5)),
            TransformWrapper(transforms.Resize((config.data.image_height, config.data.image_width), antialias=False)),
            # RandomCrop((config.data.image_height, config.data.image_width)),
            Normalize()
        ])
    )

    train_dataset, validation_dataset = random_split(court_dataset, lengths = (config.data.pct_train_split, 1.0 - config.data.pct_train_split))
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=0)             # TODO: Find out of how to get more workers on MPS, see: https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=0)  # TODO: Find out of how to get more workers on MPS, see: https://github.com/pytorch/pytorch/issues/70344

    model = TrackNet(
        in_features = config.data.n_in_features, 
        out_features = config.data.n_out_features, 
        weights_path = config.hyperparameters.path_to_weights
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adadelta(model.parameters(), lr = config.hyperparameters.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparameters.learning_rate, betas=(0.9, 0.999), weight_decay=0)

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

            optimizer.zero_grad()

            y_pred = model(x) # shape: [batch_size, output_features, image_height, image_width]

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            training_loss += loss

            if (batch_num % config.wandb.train_log_interval == config.wandb.train_log_interval - 1):
                logger.info(f"{batch_num + 1}/{len(train_dataloader)} | loss: {loss:.3f}")
                grid = make_image_grid(x, y, y_pred)
                wandb.log({
                    "epoch" : epoch,
                    "batch/train" : batch_num,
                    "loss/train"  : training_loss.item() / (batch_num + 1),
                    "images/train": wandb.Image(grid, caption="Left: Input | Middle: Labels | Right: Predicted")
                })

        # Validate
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(validation_dataloader):
                x = batch['image']
                y = batch['heatmap']

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss

                if (batch_num % config.wandb.validation_log_interval == config.wandb.validation_log_interval - 1):
                    logger.info(f"{batch_num + 1}/{len(validation_dataloader)} | val loss: {loss.item():.3f}")
                    grid = make_image_grid(x, y, y_pred)
                    wandb.log({
                        "epoch" : epoch,
                        "batch/val" : batch_num,
                        "loss/val": validation_loss.item() / (batch_num + 1),
                        "images/val" : wandb.Image(grid, caption="Left: Input | Middle: Labels | Right: Predicted")
                    })

        torch.save(model.state_dict(), f"models/model_{config.base.exp_name}_{epoch+1}epoch.pt")

    wandb.finish()



if __name__ == "__main__":
    train()

