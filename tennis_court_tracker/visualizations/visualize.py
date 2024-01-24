import torch
import hydra

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torchvision.io import read_image

from tennis_court_tracker.models.postprocess import PredictPoints, FindHomography
from tennis_court_tracker.data.court import TENNISCOURT

@hydra.main(version_base="1.2", config_path=to_absolute_path("tennis_court_tracker/conf"), config_name="config")
def main(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    ims = [read_image("data/raw/images/_7UfL2egoN0_300.png").float().to(device).unsqueeze(0)]

    for im in ims:
        pred_points = PredictPoints(in_features=config.data.n_in_features, device=device)(im)

        H, correspondance = FindHomography()(TENNISCOURT.points, pred_points[0])
        print("ASd")


if __name__ == "__main__":
    main()

