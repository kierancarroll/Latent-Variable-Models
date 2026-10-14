import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.PCA import PCAEncoder
from models.autoencoder import Autoencoder
from models.VAE import VAE
from utils import SpriteTensorDataset, plot_losses, display_samples, plot_data_encodings
from train_encoder_decoder import train_encoder_decoder_model
from VAE_loss import vae_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Training the PCA, Autoencoder and Variational Autoencoder models 
    with K = 16 latent dimensions.'''

pca_16 = PCAEncoder(16)
train_loader = DataLoader(SpriteTensorDataset(split="train"))
pca_16.fit(train_loader)

autoencoder_16 = Autoencoder(16)
train_losses, val_losses = train_encoder_decoder_model(
    model=autoencoder_16,
    loss_fn=lambda x, y: F.mse_loss(x, y, reduction='sum'),
    lr=0.001,
    epochs=12
)
plot_losses(train_losses, val_losses)

vae_16 = VAE(16)
train_losses, val_losses = train_encoder_decoder_model(
    model=vae_16,
    loss_fn=vae_loss,
    lr=0.001,
    epochs=12
)
plot_losses(train_losses, val_losses)


'''Sampling 10 random images for each of the three models.'''
samples = [pca_16.sample() for _ in range(10)]
display_samples(samples)

samples = [autoencoder_16.sample(device=device) for _ in range(10)]
display_samples(samples)

samples = [vae_16.sample(device = device) for _ in range(10)]
display_samples(samples)


'''Linearly interpolating between two points in the latent space for each of the three models.'''
train_loader = DataLoader(SpriteTensorDataset(split="train"), batch_size=20)
X = next(iter(train_loader))
display_samples(X[6].unsqueeze(0))
display_samples(X[18].unsqueeze(0))

img_left = X[6]
img_right = X[18]

def interpolate(z_start, z_end, num_steps=10):
    range = torch.linspace(0, 1, steps=num_steps)
    return [(1 - k) * z_start + k * z_end for k in range]

#PCA
z_start = pca_16.encode(img_left.unsqueeze(0))
z_end = pca_16.encode(img_right.unsqueeze(0))

z_list = interpolate(z_start.squeeze(0), z_end.squeeze(0))

decoded_imgs = [pca_16.decode(z.unsqueeze(0)).squeeze(0) for z in z_list]

display_samples(decoded_imgs)


#Autoencoder
img_left = img_left.unsqueeze(0).to(device)
img_right = img_right.unsqueeze(0).to(device)

z_start = autoencoder_16.encode(img_left)
z_end = autoencoder_16.encode(img_right)

z_list = interpolate(z_start.squeeze(0), z_end.squeeze(0))

decoded_imgs = [autoencoder_16.decode(z.unsqueeze(0)).squeeze(0) for z in z_list]

display_samples(decoded_imgs)

#VAE
mu1, logvar1 = vae_16.encode(img_left)
mu2, logvar2 = vae_16.encode(img_right)
z_start = vae_16.reparameterize(mu1, logvar1)
z_end = vae_16.reparameterize(mu2, logvar2)

z_list = interpolate(z_start.squeeze(0), z_end.squeeze(0))

decoded_imgs = [vae_16.decode(z.unsqueeze(0)).squeeze(0) for z in z_list]

display_samples(decoded_imgs)



''''Visualizing 2D latent encoding of images for each of the three models'''

pca_2 = PCAEncoder(2)
train_loader = DataLoader(SpriteTensorDataset(split="train"))
pca_2.fit(train_loader)

autoencoder_2 = Autoencoder(2)
train_losses, val_losses = train_encoder_decoder_model(
    model=autoencoder_2,
    loss_fn=lambda x, y: F.mse_loss(x, y, reduction='sum'),
    lr=0.001,
    epochs=12
)

vae_2 = VAE(2)
train_losses, val_losses = train_encoder_decoder_model(
    model=vae_2,
    loss_fn=vae_loss,
    lr=0.001,
    epochs=12
)

plot_data_encodings(pca_2.encode, SpriteTensorDataset(split="train"), device="cpu")
plot_data_encodings(autoencoder_2.encode, SpriteTensorDataset(split="train"))
plot_data_encodings(vae_2.encode, SpriteTensorDataset(split="train"))