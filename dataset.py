import os
import numpy as np
import torch
import torchvision
import cv2

from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

class EmojisDataset(Dataset):

	def __init__(self, files_dir:str='data',transform=None):
		super().__init__()
		dataset_path = os.path.join(os.getcwd(),files_dir)
		self.files = [os.path.join(dataset_path,file) for file in os.listdir(dataset_path) if file.endswith('.png')]
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		image = cv2.imread(self.files[index], cv2.IMREAD_COLOR)

		# Apply transform if necessary
		if self.transform:
			image = self.transform(image)
		else:
			image = torchvision.transforms.ToTensor()(image)

		return image


if __name__ == "__main__":
	# transform = transforms.Compose([
	# 	transforms.ToTensor(),
	# ])
	# dataset = EmojisDataset(transform=transform)
	dataset = EmojisDataset()
	
	image = dataset.__getitem__(10)
	assert type(image) == torch.Tensor, "The image should be a tensor"
	assert dataset.__getitem__(10).size() == (3,72,72), "There is an issue with the image size"