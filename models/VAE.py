import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    A Convolutional Variational Autoencoder for 32x32 RGB images.

    Args:
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.encode_mu = nn.Linear(2048,latent_dim)
        self.encode_logvar = nn.Linear(2048,latent_dim)
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
        Encodes the input image into latent mean and log-variance vectors.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32), where B is the batch size.

        Returns:
            Tuple[Tensor, Tensor]: Two tensors of shape (B, latent_dim),
            representing the mean (mu) and log-variance (logvar) of the approximate posterior.
        """
        encoder_part1 = self.encoder(x)
        mean = self.encode_mu(encoder_part1)
        log_variance = self.encode_logvar(encoder_part1)
        return (mean, log_variance)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (Tensor): Mean of the latent distribution, shape (B, latent_dim).
            logvar (Tensor): Log-variance of the latent distribution, shape (B, latent_dim).

        Returns:
            Tensor: Sampled latent vector z of shape (B, latent_dim).
        """
        sigma = torch.sqrt(torch.exp(logvar))
        z = mu + sigma*(torch.randn((logvar.shape[0],self.latent_dim), device=mu.device))
        return z


    def decode(self, z):
        """
        Decodes a latent vector into a reconstructed image.

        Args:
            z (Tensor): Latent vector of shape (B, latent_dim).

        Returns:
            Tensor: Reconstructed image tensor of shape (B, 3, 32, 32), values in [0, 1].
        """
        decoded = self.decoder(z)
        return decoded


    def forward(self, x):
        """
        Forward pass through the full VAE: encode -> reparameterize -> decode.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - recon_x (Tensor): Reconstructed image of shape (B, 3, 32, 32).
                - mu (Tensor): Latent mean of shape (B, latent_dim).
                - logvar (Tensor): Latent log-variance of shape (B, latent_dim).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

    def sample(self, z=None, device="cpu"):
        """
        Samples and decodes a latent vector into an image.

        Args:
            z (Tensor or None): Latent vector of shape (latent_dim,). If None, samples from N(0, I).
            device (str): PyTorch device the model is using. To utilize GPU, the device string may be passed in, so
                z = z.to(device) is called before decoding z.

        Returns:
            Tensor: Reconstructed image of shape (3, 32, 32)
        """
        if z is None:
            z = torch.randn(self.latent_dim)
        z = z.to(device)
        return self.decode(z.reshape((1,self.latent_dim))).reshape((3,32,32))


    def get_latent_dim(self):
        """ Returns latent dimension """
        return self.latent_dim