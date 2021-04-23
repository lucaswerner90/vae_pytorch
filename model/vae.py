import torch
import numpy as np
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder

class VariationalAutoencoder(nn.Module):
	def __init__(self, latent_dim: int = 128, image_channels: int = 3):
		super().__init__()
		self.latent_dim = latent_dim
		self.encoder = Encoder(in_channels=image_channels, latent_dim=latent_dim)
		self.decoder = Decoder(latent_dims=latent_dim, out_channels=image_channels, image_size=(72,72))

	def forward(self,x):
		latent_mu, latent_logvar = self.encoder(x)
		x = self.reparametrize(latent_mu, latent_logvar)
		x_reconstructed = self.decoder(x)
		return x_reconstructed, latent_mu, latent_logvar

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.rand_like(std)
		return mu + std * eps


def test_vae():
	vae = VariationalAutoencoder(latent_dim=128, image_channels=3)
	test_input = torch.randn(1, 3, 72, 72)
	test_output, *_ = vae(test_input)
	assert test_input.size() == test_output.size()

if __name__ == "__main__":
	test_vae()