import torch
import numpy as np
from torch import nn

class Decoder(nn.Module):
	def __init__(self, latent_dims: int, out_channels: int, image_size=(72,72)):
		super().__init__()
		self.latent_dims = latent_dims
		self.out_channels = out_channels
		self.image_size = image_size
		self.fc1 = nn.Linear(in_features=latent_dims, out_features=64*64*32)
		self.conv1 = nn.ConvTranspose2d(
			in_channels=32,
			out_channels=32,
			kernel_size=5,
			stride=1,
			padding=0
		)
		self.conv2 = nn.ConvTranspose2d(
			in_channels=32,
			out_channels=self.out_channels,
			kernel_size=5,
			stride=1,
			padding=0
		)
	def forward(self, x):
		x = self.fc1(x)
		x = nn.LeakyReLU(0.1)(x)  # (1,64*64*32)
		x = x.resize(x.size(0),32,64,64)
		x = self.conv1(x)
		x = nn.LeakyReLU(0.1)(x)  # (1,64*64*32)
		x = self.conv2(x)
		x = torch.sigmoid(x)
		y = x.reshape(x.size(0), self.out_channels, self.image_size[0], self.image_size[1])
		return y


def test_decoder():
	dimensions = 128
	image_channels = 3
	decoder = Decoder(latent_dims=dimensions,out_channels=image_channels)
	test_input = torch.randn(dimensions)
	test_input.unsqueeze_(0)
	test_output = decoder(test_input)
	test_output.squeeze_(0)
	assert test_output.size() == (image_channels,72,72)

if __name__ == "__main__":
	test_decoder()