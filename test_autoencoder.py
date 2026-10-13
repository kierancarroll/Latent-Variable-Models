import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from train_autoencoder import train_encoder_decoder_model
from utils import plot_losses, SpriteTensorDataset, display_samples

# Fitting and Sampling Autoencoder Model
autoencoder_2 = Autoencoder(2)

train_losses, val_losses = train_encoder_decoder_model(
    model=autoencoder_2,
    loss_fn=lambda x, y: F.mse_loss(x, y, reduction='sum'),
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

Z = autoencoder_2.encode(X)
X_recon = autoencoder_2.decode(Z)

display_samples(X)
for z in Z:
    print(f"{z[0]:.2f}, {z[1]:.2f}", end="")
display_samples(X_recon)

print('Generating new images:')
samples = [autoencoder_2.sample(device=device) for _ in range(10)]
display_samples(samples)