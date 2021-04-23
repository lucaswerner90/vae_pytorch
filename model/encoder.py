import torch
import numpy as np
from torch import nn

class Encoder(nn.Module):
	def __init__(self, in_channels: int = 3, latent_dim: int = 128):
		super().__init__()
		self.conv1 = nn.Conv2d(
			in_channels=in_channels,
			out_channels=32,
			kernel_size=5,
			stride=1,
			padding=0
		) # input size 72 --> ouput size --> (72 - 5 + 1) = 68 x 32 filters
		self.conv2 = nn.Conv2d(
			in_channels=32,
			out_channels=32,
			kernel_size=5,
			stride=1,
			padding=0
		) # input size 68 --> ouput size --> (68 - 5 + 1) / 2 = 64 x 32 filters
		# calculates the mean vector
		self.fc_mu = nn.Linear(in_features=64*64*32, out_features=latent_dim)
		# calculate the variance vector
		self.fc_logvar = nn.Linear(in_features=64*64*32, out_features=latent_dim)

	def forward(self,x):
		x = self.conv1(x)
		x = nn.LeakyReLU(0.1)(x) # (1,32,68,68)
		x = self.conv2(x)
		x = nn.LeakyReLU(0.1)(x) # (1,32,64,64)
		x = x.reshape(x.size(0), -1)  # (1,32*64*64)
		mu = nn.LeakyReLU(0.1)(self.fc_mu(x))
		logvar = nn.LeakyReLU(0.1)(self.fc_logvar(x))
		return mu, logvar


def test_encoder():
	image_channels = 3
	image_size = (image_channels,72,72)
	dimensions = 128
	encoder = Encoder(in_channels=image_channels, latent_dim=dimensions)
	test_input = torch.randn(image_size)
	test_input.unsqueeze_(0)
	test_mu, test_logvar = encoder(test_input)
	assert test_mu.shape == (1,dimensions)
	assert test_logvar.shape == (1,dimensions)

if __name__ == "__main__":
	test_encoder()