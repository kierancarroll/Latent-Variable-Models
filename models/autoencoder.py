import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        """
        A standard convolutional autoencoder for 32x32 RGB images.

        Args:
            latent_dim (int): Dimensionality of the bottleneck (latent space).
        """
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048,latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1),
        )


    def encode(self, x):
        """
        Encodes input image into latent vector using conv layers and tanh bottleneck.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32)

        Returns:
            Tensor: Latent representation of shape (B, latent_dim), bounded in [-1, 1]
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes latent vector into reconstructed image.

        Args:
            z (Tensor): Latent tensor of shape (B, latent_dim)

        Returns:
            Tensor: Reconstructed image of shape (B, 3, 32, 32)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full autoencoder forward pass: encode -> decode.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32)

        Returns:
            Tensor: Reconstructed image of shape (B, 3, 32, 32)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def sample(self, z=None, device="cpu"):
        """
        Samples and decodes a latent vector into an image.

        Args:
            z (Tensor or None): Latent vector of shape (latent_dim,). If None, samples from Uniform(-1, 1).
            device (str): PyTorch device the model is using. To utilize GPU, the device string may be passed in, so
                z = z.to(device) is called before decoding z.

        Returns:
            Tensor: Reconstructed image of shape (3, 32, 32)
        """
        if z is None:
          z = torch.rand(self.latent_dim) * 2 - 1
        z = z.to(device)
        return self.decode(z.reshape((1, self.latent_dim))).reshape((3, 32, 32))


    def get_latent_dim(self):
        """ Returns latent dimension """
        return self.latent_dim
