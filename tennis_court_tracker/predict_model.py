import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from tennis_court_tracker.data.TennisCourtDataset import TennisCourtDataset, Rescale, ToTensor
from tennis_court_tracker.models.model import TrackNet


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
    return torch.cat([model(batch['image']) for batch in dataloader], 0)


if __name__ == "__main__":

    # Load model
    model_path = "model.pt"
    model = TrackNet(in_features = 3)
    model.load_state_dict(torch.load(model_path))

    dataset = TennisCourtDataset(
        "data/processed/keypoints.csv", 
        "data/processed/images",
        "data/processed/labels",
        transform = transforms.Compose([
            Rescale((360, 640)),
            ToTensor()
        ])
    )
    n = 1
    dataset_subset = torch.utils.data.Subset(dataset, range(n))
    dataloader = DataLoader(dataset_subset, batch_size=1, shuffle=False, num_workers=0)

    predictions = predict(model, dataloader)
    pred = predictions[0].squeeze().argmax(dim=0).detach().numpy()
    print(np.sum(0 < pred))
    plt.imshow(pred)
    plt.show()


