import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        with open(os.path.join(self.path, 'meta.json'), 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    # Load the input and target images
    def __getitem__(self, index):

        input_image = Image.open(os.path.join(self.path, self.data['pairs'][index]['image']))
        target_image = Image.open(os.path.join(self.path, self.data['pairs'][index]['labels']))

        # Apply transformations if provided
        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

    def __len__(self):
        return len(self.data['pairs'])
