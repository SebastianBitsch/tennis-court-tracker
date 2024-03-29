import json

import torch
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class TennisCourtDataset(Dataset):
    """ Tennis court dataset, adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, annotations_file_path:str, images_dir:str, heatmap_point_radius:int = 5, device:str = 'cpu', transform = None) -> None:
        """  """
        self.images_dir = images_dir
        self.transform = transform
        self.device = device
        self.heatmap_point_radius = heatmap_point_radius

        with open(annotations_file_path) as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx:int) -> dict:
        """ Get the next image and heatmap """
        image_id = self.annotations[idx]['id']
        image_path = f"{self.images_dir}/{image_id}.png"
        
        image = read_image(image_path).to(self.device).float()
        heatmap = generate_heatmap(image.shape[1:], self.annotations[idx]['kps'], point_radius=self.heatmap_point_radius, device=self.device)

        if self.transform:
            output = self.transform({"image" : image, "heatmap" : heatmap})
            image = output['image']
            heatmap = output['heatmap']

        return {
            "image" : image,
            "heatmap" : heatmap
        }


class TransformWrapper:
    """ Wraps a transform that operates on only the sample. See: https://stackoverflow.com/a/75723566/19877091 """
    def __init__(self, transform: object):
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        sample['image']   = self.transform(sample['image'])
        sample['heatmap'] = self.transform(sample['heatmap'])
        return sample


class Normalize(object):
    """ """
    def __init__(self, interval: tuple[float, float] = (0, 1)) -> None:
        assert interval[0] == 0, "Normalize to other ranges than [0,x] is not supported yet"
        self.interval = interval

    def __call__(self, sample: dict) -> dict:
        for name in sample.keys():
            # Get to range 0-1 
            # TODO: I hate this - do we even really need to normalize / can it be done earlier (?)
            sample[name] -= sample[name].min()
            max = sample[name].max()
            if 0 < max:
                sample[name] /= max
            else:
                print("Image with no keypooints - i.e. target pure zeros")
            # Scale to proper range
            if self.interval[1] != 1:
                sample[name] *= self.interval[1]
            # ... TODO: Add support for that ...

        return sample


class RandomAffine(object):
    """ """
    def __init__(self, im_size: tuple[int,int], degrees:tuple[int,int] = (-15, 15), translate:tuple[int,int] = (0.2, 0.2), scale:tuple[int,int]=(0.5, 1.5), shears:tuple[int,int]=None) -> None:
        self.im_size = im_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shears = shears

    def __call__(self, sample: dict) -> dict:
        i,j,k,l = transforms.RandomAffine.get_params(degrees = self.degrees, translate = self.translate, scale_ranges=self.scale, shears=self.shears, img_size=self.im_size)

        sample['image'] = F.affine(sample['image'], i, j, k, l)
        sample['heatmap'] = F.affine(sample['heatmap'], i, j, k, l)
        return sample


class RandomCrop(object):
    """
    Custom random crop function to get the same random cropping for both the image and the heatmap. 
    Using the built in RandomCrop function gives different croppings for the images
    """
    def __init__(self, output_size: int | tuple[int,int]) -> None:
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        i, j, h, w = transforms.RandomCrop.get_params(sample['image'], output_size=self.output_size)

        sample['image'] = F.crop(sample['image'], i, j, h, w)
        sample['heatmap'] = F.crop(sample['heatmap'], i, j, h, w)
        return sample


def gaussian_kernel(radius:int, sigma:int):
    """Generates a Gaussian kernel. Centered at the middle"""
    x = torch.arange(-radius, radius + 1)
    x = (1.0 / (2 * torch.pi * sigma)) * torch.exp(-(((x - 0)** 2 + (x - 0)**2) / (2 * sigma))) * (2 * torch.pi * sigma)
    return torch.outer(x, x)


def is_kernel_in_image(image_size: tuple[int,int], point: tuple[int,int], border_size:int) -> bool:
    return 0 <= point[0] - border_size and 0 <= point[1] - border_size and point[0] + border_size < image_size[0] and point[1] + border_size < image_size[1]


def generate_heatmap(size: tuple[int,int], keypoints:list, point_radius:int, device:str="cpu") -> torch.FloatTensor:
    """
    Generate a "heatmap" of points on an image. 
    Every point will be represented by a gaussian on the image
    returns a uint8 array of values in range 0-255
    not all keypoints are sure to be in the image, especially true when image is cropped etc
    """
    gaussian = gaussian_kernel(point_radius, point_radius*point_radius) # NOTE: We just use radius**2 as the sigma, thats good enough for this and one less param to tune
    heatmap = torch.zeros((1, *size), dtype=torch.float32, device=device)
    
    for (cx, cy) in keypoints:
        if not is_kernel_in_image(size, (cy, cx), point_radius):
            continue
        heatmap[0, cy-point_radius:cy+point_radius+1, cx-point_radius:cx+point_radius+1] = gaussian # TODO: Point could overlap tbh

    return heatmap
