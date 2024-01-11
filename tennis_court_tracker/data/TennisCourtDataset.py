import json

import torch

from torch.utils.data import Dataset
from torchvision.io import read_image


class TennisCourtDataset(Dataset):
    """ Tennis court dataset, adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, annotations_file_path:str, images_dir:str, device:str = 'cpu', transform = None) -> None:
        """  """
        self.images_dir = images_dir
        self.transform = transform
        self.device = device
        with open(annotations_file_path) as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx:int) -> dict:
        """ Get the next image and heatmap """
        image_id = self.annotations[idx]['id']
        image_path = f"{self.images_dir}/{image_id}.png"
        
        image = read_image(image_path)
        heatmap = generate_heatmap(image.shape[1:], self.annotations[idx]['kps'])

        if self.transform:
            image = self.transform(image)
            heatmap = self.transform(heatmap)

        return {
            "image" : image.float().to(self.device), 
            "heatmap" : heatmap.squeeze().to(self.device)
        }


def gaussian_kernel(size:int, sigma2:int):
    """Generates a Gaussian kernel. Centered at the middle"""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = (1.0 / (2 * torch.pi * sigma2)) * torch.exp(-(((x - 0)** 2 + (x - 0)**2) / (2 * sigma2))) * (2 * torch.pi * sigma2)
    kernel = torch.outer(x, x)
    return kernel

def point_in_image(image_size: tuple[int,int], point: tuple[int,int], border_size:int):
    return 0 <= point[0] - border_size and 0 <= point[1] - border_size and point[0] + border_size < image_size[0] and point[1] + border_size < image_size[1]

def generate_heatmap(size: tuple[int,int], keypoints:list, radius:int = 5, sigma2:int = 10) -> torch.FloatTensor:
    """
    Generate a "heatmap" of points on an image. 
    Every point will be represented by a gaussian on the image
    returns a uint8 array of values in range 0-255
    not all keypoints are sure to be in the image, especially true when image is cropped etc
    """
    heatmap = torch.zeros((1, *size), dtype=torch.uint8)
    gaussian = (255 * gaussian_kernel(2 * radius, sigma2)).to(torch.uint8)
    
    for cx, cy in keypoints:
        if not point_in_image(size, (cy, cx), radius):
            continue
        heatmap[0, cy-radius:cy+radius, cx-radius:cx+radius] = gaussian

    return heatmap