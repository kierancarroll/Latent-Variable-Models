import torch

class PCAEncoder:
    def __init__(self, latent_dim):
        """
        PCA-based encoder using SVD, implemented in PyTorch.

        Args:
            latent_dim (int): Number of principal components to keep.
        """
        self.latent_dim = latent_dim

        # Values to be determined in fit()
        self.mean = None                      # Tensor of shape (1, M)
        self.principal_components = None      # Tensor of shape (M, latent_dim)
        # where M = 3 * 32 * 32

    def fit(self, dataloader):
        """
        Fits the PCA model using SVD on data from the dataloader.

        Sets self.mean and self.principal_components - You need to put the mean and principal_components
        into these variables to pass the autograder.

        Args:
            dataloader (DataLoader): Loader providing images of shape (B, 3, 32, 32)
            device (str): Device for computation.
        """
        images = []
        for batch in dataloader:
            images.append(batch)
        images = torch.cat(images, dim=0)
        images_flattened = torch.flatten(images, start_dim=1, end_dim=3)
        images_mean = torch.mean(images_flattened, dim=0, keepdim=True)
        images_centered = images_flattened - images_mean

        U, S, VT = torch.linalg.svd(images_centered)

        principal_components = VT.T[:,:self.latent_dim]

        self.principal_components = principal_components
        self.mean = images_mean

    def encode(self, x):
        """
        Encodes input images using the learned PCA projection.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32)

        Returns:
            Tensor: Encoded latent tensor of shape (B, latent_dim)
        """
        B = x.size(0)
        x_flattened = torch.flatten(x, start_dim=1, end_dim=3)
        x_centered = x_flattened - self.mean

        encoded_data = x_centered @ self.principal_components
        return encoded_data


    def decode(self, z):
        """
        Decodes latent vectors back to the original image space.

        Args:
            z (Tensor): Latent tensor of shape (B, latent_dim)

        Returns:
            Tensor: Reconstructed image tensor of shape (B, 3, 32, 32)
        """
        B = z.size(0)

        PC_T = self.principal_components.T
        decoded_data = z @ PC_T
        decoded_data = decoded_data + self.mean
        decoded_data_reshaped = decoded_data.reshape(B, 3, 32, 32)
        return decoded_data_reshaped

    def sample(self, z=None):
        """
        Samples and decodes a latent vector into an image.

        Args:
            z (Tensor or None): Latent vector of shape (latent_dim,). If None, samples from N(0, I).

        Returns:
            Tensor: Reconstructed image of shape (3, 32, 32)
        """
        if z is None:
            z = torch.randn(self.latent_dim)
        return self.decode(z.reshape((1, self.latent_dim))).reshape((3, 32, 32))


    def get_latent_dim(self):
        """ Returns latent dimension """
        return self.latent_dim