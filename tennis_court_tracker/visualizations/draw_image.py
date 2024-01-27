import random
import os

import torch
import hydra
import cv2

import matplotlib.pyplot as plt

import numpy as np
from torchvision import transforms

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torchvision.io import read_image

from tennis_court_tracker.models.postprocess import PredictPoints, FindHomography
from tennis_court_tracker.data.court import TENNISCOURT

@hydra.main(version_base="1.2", config_path=to_absolute_path("tennis_court_tracker/conf"), config_name="config")
def main(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    weights_path = "models/model_affine3_45epoch.pt"

    n = 4

    file_names = [random.choice(os.listdir("data/raw/images")) for _ in range(n*n)]
    ims = [read_image(f"data/raw/images/{fn}").float().to(device).unsqueeze(0) for fn in file_names]

    fig, axs = plt.subplots(n, n)
    axs = axs.flatten()

    for i, im in enumerate(ims):
        pred_points = PredictPoints(in_features=config.data.n_in_features, weights_path=weights_path, device=device)(im)[0]

        H, correspondance = FindHomography()(TENNISCOURT.points, pred_points)
        
        axs[i].imshow(transforms.Resize((360, 640), antialias=False)(im).squeeze().permute(1, 2, 0).cpu().numpy() / 255, interpolation='none')
        tf = cv2.perspectiveTransform(TENNISCOURT.points.reshape(1, -1, 2), np.linalg.inv(H)).squeeze()


        for (a, b) in TENNISCOURT.lines.values():
            ta = tf[a]
            tb = tf[b]
            axs[i].plot([ta[0], tb[0]], [ta[1], tb[1]], '-', c='orangered', linewidth=1)
    
    for ax in axs.ravel():
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

