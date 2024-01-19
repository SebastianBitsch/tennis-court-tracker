import torch
import wandb
import hydra
import logging

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, RandomCrop, TransformWrapper
from tennis_court_tracker.models.model import TrackNet

logger = logging.getLogger(__name__)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch['image']) for batch in dataloader], 0)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    model = TrackNet(in_features = config.data.n_in_features).to(device)
    # Load model weights
    if config.hyperparameters.continue_training_from_weights:
        logger.info(f"Loading model state dict at: {config.hyperparameters.path_to_weights}")
        model.load_state_dict(torch.load(config.hyperparameters.path_to_weights))


    dataset = TennisCourtDataset(
        annotations_file_path = to_absolute_path("data/raw/data_train.json"), 
        images_dir = to_absolute_path("data/raw/images"),
        device = device,
        transform = transforms.Compose([
            TransformWrapper(transforms.Resize((config.data.image_height, config.data.image_width), antialias=True))
            # RandomCrop((config.data.image_height, config.data.image_width)),
        ])
    )
    n = 3
    dataset_subset = torch.utils.data.Subset(dataset, range(n))
    dataloader = DataLoader(dataset_subset, batch_size=1, shuffle=False, num_workers=0)

    predictions = predict(model, dataloader)
    pred = predictions.argmax(dim=1)


if __name__ == "__main__":
    main()
