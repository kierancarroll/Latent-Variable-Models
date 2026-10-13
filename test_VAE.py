import torch
from torch.utils.data import DataLoader
from models.VAE import VAE
from train_encoder_decoder import train_encoder_decoder_model
from VAE_loss import vae_loss
from utils import plot_losses, SpriteTensorDataset, display_samples

# Train VAE
vae_2 = VAE(2)

train_losses, val_losses = train_encoder_decoder_model(
    model=vae_2,
    loss_fn=vae_loss,
    lr=0.001,
    epochs=12
)
plot_losses(train_losses, val_losses)

# See how well the autoencoder works on the training set
train_loader = DataLoader(SpriteTensorDataset(split="train"), batch_size=10)
X = next(iter(train_loader))

# Put input on GPU if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)

Z_mu, Z_logvar  = vae_2.encode(X)
X_recon = vae_2.decode(Z_mu)

display_samples(X)
for z in Z_mu:
    print(f"    {z[0]:.2f}, {z[1]:.2f}", end="    ")
display_samples(X_recon)

print()
print()
print('Generating new images:')
samples = [vae_2.sample(device=device) for _ in range(10)]
display_samples(samples)