import torch
import numpy as np
import os
import torchvision
import time

from PIL import Image
from dataset import EmojisDataset
from model.vae import VariationalAutoencoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_file = os.path.join(dir_path, 'checkpoint.pkl')


dataset = EmojisDataset()
dataset_len = len(dataset.files)
train_dataset_len = int(dataset_len * 0.8)
test_dataset_len = dataset_len - train_dataset_len


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_dataset_len, test_dataset_len])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)


class VAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_recon, target, mu, logvar):
        data_loss = torch.nn.functional.mse_loss(x_recon, target, reduction='sum')
        kl_div = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return data_loss + kl_div



def save_model(model, latent_dim, optimizer, epochs, file):
	checkpoint_dict = {
        'model': model.state_dict(),
		'epochs': epochs,
		'latent_dim': latent_dim,
		'optimizer': optimizer.state_dict()
    }
	torch.save(checkpoint_dict, file)



def train_model(model, epochs, optimizer):
	criterion = VAELoss()	
	model.to(device)
	model.train()
	for i in range(epochs):
		avg_loss = []
		start_time = time.time()
		for images in train_dataloader:
			images = images.to(device)
			
			optimizer.zero_grad()

			output, mu, logvar = model(images)
			loss = criterion(output, images, mu, logvar)
			
			loss.backward()
			optimizer.step()

			avg_loss.append(loss.item())

		print('Loss on epoch {} \t ==> {:10.5f} \t took {:4.2f}s'.format(i + 1, np.mean(avg_loss),time.time()-start_time))

	return model


def eval_model(model):
	model.eval()
	model.to(device)
	criterion = VAELoss()
	valid_losses = []
	for images in test_dataloader:
		images = images.to(device)
		output, mu, logvar = model(images)
		loss = criterion(output, images, mu, logvar)
		valid_losses.append(loss.item())
	
	return np.mean(valid_losses)

def generate_images(model):
	images = next(iter(test_dataloader))
	images = images.to(device)
	model.eval()
	model.to(device)
	with torch.no_grad():
		generated, *_ = model(images)
		for i in range(generated.size(0)):
			image = generated[i]
			image = image.clamp(0, 1)
			image = torchvision.transforms.ToPILImage()(image)
			image.save(os.path.join(dir_path,'generated',f'generated_{i}.png'))


def generate_one_image(model, index):
	model.eval()
	with torch.no_grad():
		item = test_dataset.__getitem__(index)
		item.requires_grad = False
		item.unsqueeze_(0)
		output, *_ = model(item)
		output.squeeze_(0)
		output = output.clamp(0, 1)
		output = torchvision.transforms.ToPILImage()(output)
		output.save(os.path.join(dir_path,f'generated_one_{index}.png'))

if __name__ == "__main__":
	load_from_checkpoint = False
	checkpoint_file = os.path.join(dir_path, 'checkpoint.pkl')
	checkpoint = torch.load(checkpoint_file) if os.path.exists(checkpoint_file) else None
	latent_dim = 32
	epochs = 10
	model = VariationalAutoencoder(latent_dim=latent_dim, image_channels=3)
	optim = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	if load_from_checkpoint:
		model.load_state_dict(checkpoint['model'])
		optim.load_state_dict(checkpoint['optimizer'])
	else:
		model = train_model(model=model,epochs=epochs, optimizer=optim)
		save_model(model, latent_dim, optim, epochs, checkpoint_file)

	eval_loss = eval_model(model)
	print('Eval loss of the model {:10.4f}'.format(eval_loss))
	generate_one_image(model,50)
