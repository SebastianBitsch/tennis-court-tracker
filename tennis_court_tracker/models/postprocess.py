import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms as TF
from tennis_court_tracker.models.tracknet import TrackNet

from scipy.optimize import linear_sum_assignment

class PredictPoints(nn.Module):
    """ """

    def __init__(self, in_features:int, weights_path: str | None = None, device:str="cpu") -> None:
        """ """
        super(PredictPoints, self).__init__()

        self.pipeline = nn.Sequential(
            TrackNet(in_features = in_features, out_features=1, weights_path=weights_path), # TODO: out_features might need to change
            TF.ConvertImageDtype(torch.uint8),
            nn.Threshold(130, 0), # TODO: Threshold should maybe be dynamic (?)
            TF.GaussianBlur(kernel_size = (9,9), sigma = 2)
        ).to(device)

    def forward(self, im: torch.Tensor) -> list:

        with torch.no_grad():
            pred_heatmaps = self.pipeline(im).squeeze(1) # dim: [batchsize, out_features, h, w]

        # TODO: Iterate over the batch. I dont think this can be parallelized nicely
        predicted_points = []
        for heatmap in pred_heatmaps.cpu().numpy():
            pred_points = cv2.HoughCircles(
                heatmap,
                cv2.HOUGH_GRADIENT,
                dp=1,           # Inverse ratio of accumulator resolution to image resolution
                minDist=20,     # Minimum distance between the centers of the detected circles
                param1=10,      # Upper threshold for the edge detector
                param2=3,       # Threshold for center detection
                minRadius=1,    # Minimum radius of the detected circles
                maxRadius=20    # Maximum radius of the detected circles
            )
            predicted_points.append(pred_points.squeeze()[:, :2])
        
        return predicted_points


class FindHomography(torch.nn.Module):
    """ """
    def __init__(self) -> None:
        """ NOTE: This doesnt work for batched input, expects just a single set of points. Not really a torch module"""
        super(FindHomography, self).__init__()

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        return points - np.mean(points, axis=0)

    def forward(self, court_points: np.ndarray, pred_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ """
        norm_court = self.normalize_points(court_points)
        norm_pred = self.normalize_points(pred_points)

        distances_matrix = np.array([ np.linalg.norm(norm_pred - p, axis=1) for p in norm_court])

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        court_idxs, pred_idxs, = linear_sum_assignment(distances_matrix)

        H, _ = cv2.findHomography(
            pred_points[pred_idxs, :], 
            court_points[court_idxs, :], 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0, 
            maxIters=10000
        )
        return H, pred_idxs