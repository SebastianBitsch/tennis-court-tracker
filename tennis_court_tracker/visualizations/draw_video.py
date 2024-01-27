import torch
import hydra
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    fps = 1000 / 60
    n_frames = 1500
    starting_frame = 36

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()

    def draw_frame(i):
        ax.clear() # Clear the previous lines
    
        file_name =f"frame{starting_frame+i}.jpg"
        im = read_image(f"data/videos/video6/{file_name}").float().to(device).unsqueeze(0)

        # Get homography matrix
        pred_points = PredictPoints(in_features=config.data.n_in_features, weights_path=weights_path, device=device)(im)[0]
        H, _ = FindHomography()(TENNISCOURT.points, pred_points)

        ax.imshow(transforms.Resize((config.data.image_height, config.data.image_width), antialias=False)(im).squeeze().permute(1, 2, 0).cpu().numpy() / 255, interpolation='none')
        tf = cv2.perspectiveTransform(TENNISCOURT.points.reshape(1, -1, 2), np.linalg.inv(H)).squeeze()

        for (a, b) in TENNISCOURT.lines.values():
            ta = tf[a]
            tb = tf[b]
            ax.plot([ta[0], tb[0]], [ta[1], tb[1]], '-', c='orangered', linewidth=2)

        return ax

    ani = animation.FuncAnimation(fig, draw_frame, interval=fps, frames=n_frames)
    ani.save('movie6.mp4')



if __name__ == "__main__":
    main()

