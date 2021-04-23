import unittest
import torch
from dataset import EmojisDataset

class DatasetSpec(unittest.TestCase):
	
	def contains_files(self):
		dataset = EmojisDataset()
		has_files = dataset.__len__() > 0
		self.assertTrue(has_files, True)

	def check_image_is_tensor(self):
		dataset = EmojisDataset()
		image = dataset.__getitem__(10)
		self.assertTrue(type(image), torch.Tensor)
	
	def check_image_size(self):
		dataset = EmojisDataset()
		image = dataset.__getitem__(10)
		self.assertTrue(image.size(), (72,72,72))