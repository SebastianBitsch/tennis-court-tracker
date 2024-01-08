import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize

class TennisCourtDataset(Dataset):
    """Tennis court dataset, adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, keypoints_file:str, images_dir:str, transform=None) -> None:
        """
        Arguments:
            keypoints_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.keypoints_df = pd.read_csv(keypoints_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.keypoints_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.keypoints_df.iloc[idx]['image_name'])
        image = np.load(img_name)
        keypoints = self.keypoints_df.iloc[idx, 2:]
        keypoints = np.array([keypoints], dtype=float).reshape(-1, 2)
        
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w, 3))

        # h and w are swapped for keypoints because for images,
        # x and y axes are axis 1 and 0 respectively
        keypoints = keypoints * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': keypoints}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h, left: left + new_w]

        keypoints = keypoints - [left, top]

        return {'image': image, 'keypoints': keypoints}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {
            'image': torch.Tensor(image),
            'keypoints': torch.Tensor(keypoints)
        }