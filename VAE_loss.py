import torch
import torch.nn.functional as F

def recon_loss(output, x):
    """
    Computes the reconstruction loss for a Variational Autoencoder (VAE)

    Args:
        output (Tuple[Tensor, Tensor, Tensor]):
            - recon_x: Reconstructed output from the decoder, shape (B, C, H, W)
            - mu: Mean of the approximate posterior, shape (B, latent_dim)
            - logvar: Log-variance of the approximate posterior, shape (B, latent_dim)
        x (Tensor): Original input image, shape (B, C, H, W)

    Returns:
        Tensor: Scalar reconstruction loss value
    """
    recon_x, mu, logvar = output
    loss = F.mse_loss(recon_x, x, reduction='sum')
    return loss


def kl_div(output, x):
    """
    Computes the KL divergence component of the loss for a Variational Autoencoder (VAE)

    Args:
        output (Tuple[Tensor, Tensor, Tensor]):
            - recon_x: Reconstructed output from the decoder, shape (B, C, H, W)
            - mu: Mean of the approximate posterior, shape (B, latent_dim)
            - logvar: Log-variance of the approximate posterior, shape (B, latent_dim)
        x (Tensor): Original input image, shape (B, C, H, W)

    Returns:
        Tensor: Scalar KL divergence
    """
    recon_x, mu, logvar = output
    KL_Divergence = -0.5*torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
    return KL_Divergence

def vae_loss(output, x):
    """
    Computes the loss for a Variational Autoencoder (VAE), consisting of
    reconstruction loss and KL divergence.

    Args:
        output (Tuple[Tensor, Tensor, Tensor]):
            - recon_x: Reconstructed output from the decoder, shape (B, C, H, W)
            - mu: Mean of the approximate posterior, shape (B, latent_dim)
            - logvar: Log-variance of the approximate posterior, shape (B, latent_dim)
        x (Tensor): Original input image, shape (B, C, H, W)

    Returns:
        Tuple[Tensor, Tensor, Tensor):
            - Scalar loss value (sum of reconstruction loss and KL divergence)
            - Tensor: Scalar reconstruction loss
            - Tensor: Scalar KL divergence
    """
    recon = recon_loss(output, x)
    kl = kl_div(output, x)
    return recon + kl, recon, kl
