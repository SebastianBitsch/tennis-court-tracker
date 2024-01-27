import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms as TF
from tennis_court_tracker.models.tracknet import TrackNet

from scipy.optimize import linear_sum_assignment
from tennis_court_tracker.data.court import TENNISCOURT

from skimage.morphology import erosion
from skimage.morphology import disk 


class PredictPoints(nn.Module):
    """ """

    def __init__(self, in_features:int, weights_path: str | None = None, device:str="cpu") -> None:
        """ """
        super(PredictPoints, self).__init__()

        self.pipeline = nn.Sequential(
            TrackNet(in_features = in_features, out_features=1, weights_path=weights_path), # TODO: out_features might need to change
            TF.ConvertImageDtype(torch.uint8),
            nn.Threshold(254, 0), # TODO: Threshold should maybe be dynamic (?)
        ).to(device)


    def _detect_points(self, binary_image: np.ndarray) -> np.ndarray:
        """ """
        # We have to invert the image, so dumb - but doesnt seem to work without
        inv_binary = cv2.bitwise_not(binary_image)

        params = cv2.SimpleBlobDetector_Params()         
        # TODO: We can filter out points on all sorts of parameters. See: https://stackoverflow.com/a/28573944/19877091

        detector = cv2.SimpleBlobDetector_create(params) 
        pred_points = cv2.KeyPoint_convert(detector.detect(inv_binary))

        return pred_points


    def forward(self, im: torch.Tensor) -> list:

        with torch.no_grad():
            pred_heatmaps = self.pipeline(im).squeeze(1) # dim: [batchsize, out_features, h, w] -> [batchsize, h, w]

        predicted_points = []
        # TODO: Iterate over the batch. I dont think this can be parallelized nicely
        for heatmap in pred_heatmaps.cpu().numpy():
            
            for erosion_amount in range(5):
                binary_im = erosion(heatmap, disk(erosion_amount))
                pred_points = self._detect_points(binary_im)
                
                # We want to keep removing points until we have a reasonable amount to get rid of noise
                if len(pred_points) < 1.5 * len(TENNISCOURT.points):
                    print(f"Found binary image with {len(pred_points)} points at erosion({erosion_amount})")
                    predicted_points.append(pred_points)
                    break

        return predicted_points


class FindHomography(torch.nn.Module):
    """ """
    def __init__(self) -> None:
        """ NOTE: This doesnt work for batched input, expects just a single set of points. Not really a torch module just pretending to be"""
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
            maxIters=1000000
        )
        return H, pred_idxs