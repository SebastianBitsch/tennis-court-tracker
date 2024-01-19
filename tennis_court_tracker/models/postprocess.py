import torch
import cv2

import numpy as np
from torchvision import transforms

from scipy.optimize import linear_sum_assignment


class PredictedPoints(torch.nn.Module):
    """ """
    def __init__(self, threshold_value: int = 130, gaussian_kernel_size: int = 9, gaussian_sigma: float = 2.0) -> None:
        """ """
        super(PredictedPoints, self).__init__()
        self.threshold = torch.nn.Threshold(threshold_value, 0)
        self.blur = transforms.GaussianBlur(kernel_size = gaussian_kernel_size, sigma = gaussian_sigma)

    def forward(self, x: torch.Tensor) -> list[np.ndarray]:
        """ Takes as input a batch of binary images and returns a list of points"""
        x = x.to(torch.uint8)
        x = self.threshold(x)
        x = self.blur(x)
        
        # TODO: I dont think there is a nice way to parallelize this unfortunately
        predicted_points = []
        for heatmap in x.cpu().numpy():
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
            predicted_points.append(pred_points[0, :, :2])
        
        return predicted_points


class FindHomography(torch.nn.Module):
    """ """
    def __init__(self) -> None:
        """ """
        super(FindHomography, self).__init__()
        
    def forward(self, true_points: np.ndarray, pred_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ """
        norm_true = normalize_points(true_points)
        norm_pred = normalize_points(pred_points)

        # calculate a n x m distance matrix between predicted points and court points
        distances_matrix = np.array([ np.linalg.norm(norm_pred - p, axis=1) for p in norm_true])

        # Calculate the cheapest way of arranging the point, sort of like bipartite matching
        true_idxs, pred_idxs, = linear_sum_assignment(distances_matrix)

        H, _ = cv2.findHomography(pred_points[pred_idxs, :], true_points[true_idxs, :], method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000)
        return H, pred_idxs
    

def normalize_points(points: np.ndarray) -> np.ndarray:
    return points - np.mean(points, axis=0)
