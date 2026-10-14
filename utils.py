import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class SpriteTensorDataset(Dataset):
    def __init__(self, root_dir="tensor_data", split="train"):
        """
        Args:
            root_dir (str): Path to the directory containing the .pt files.
            split (str): Either 'train' or 'test' to load from the respective .pt file.
            transform (callable, optional): Optional transform to apply to each image.
        """
        assert split in ["train", "validation"], "split must be 'train' or 'validation'"
        tensor_path = os.path.join(root_dir, f"{split}.pt")
        
        self.data = torch.load(tensor_path)  
        
        self.transform = lambda x: x.float() / 255.0  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)  
        return image 

def plot_data_encodings(encode_fn, dataset, batch_size=256, device=None):
    """
    Encodes and plots 2D representations of dataset points using the given encoding function.
    All points are drawn as very small dots to visualize density.

    Args:
        encode_fn (Callable): Function that takes a batch of images and returns:
                              either (B, 2) or tuple (mu, logvar) of shape (B, 2).
        dataset (Dataset): Dataset of image tensors.
        batch_size (int): Batch size for encoding. Default is 256.
        device (str or torch.device): Device for computation. Defaults to auto-detect.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_encodings = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)

            z = encode_fn(inputs)
            if isinstance(z, tuple):
                z = z[0]  # use mean for VAE

            all_encodings.append(z.cpu())

    points = torch.cat(all_encodings, dim=0).numpy()

    # Compute square bounds based on central 80% of points
    lower = np.percentile(points, 10, axis=0)
    upper = np.percentile(points, 90, axis=0)
    center = (upper + lower) / 2
    radius = max((upper - lower) / 2)
    max_range = np.max(np.abs(points - center))
    side = max(2 * max_range, radius * 2)
    half_side = side / 2

    plt.figure(figsize=(6, 6))
    plt.scatter(
        points[:, 0], points[:, 1],
        s=0.5,
        alpha=0.3,
        c='black',
        linewidths=0
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(center[0] - half_side, center[0] + half_side)
    plt.ylim(center[1] - half_side, center[1] + half_side)
    plt.title("2D Encodings of Dataset")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_samples(samples):
    """
    Displays a list of image tensors in a horizontal row.

    Args:
        samples (List[Tensor]): List of image tensors, each of shape (3, H, W).
    """
    num_samples = len(samples)
    _, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    
    if num_samples == 1:
        axes = [axes] 

    for i, ax in enumerate(axes):
        img = samples[i].detach().cpu().numpy()
        img = img.transpose(1, 2, 0)  
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (List[float]): List of average training losses per epoch.
        val_losses (List[float]): List of average validation losses per epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 5))

    # if the losses are tuples or lists plot all of them
    if isinstance(train_losses[0], (list, tuple)):
        train_recon_kl = [loss[0] for loss in train_losses]
        train_recon = [loss[1] for loss in train_losses]
        train_kl = [loss[2] for loss in train_losses]
        val_recon_kl = [loss[0] for loss in val_losses]
        val_recon = [loss[1] for loss in val_losses]
        val_kl = [loss[2] for loss in val_losses]

        plt.plot(epochs, train_recon_kl, label='Train Loss')
        # Get color of the above line
        color = plt.gca().lines[-1].get_color()
        plt.plot(epochs, train_recon, label='Train Recon Loss', linestyle='--', color=color)
        plt.plot(epochs, train_kl, label='Train KL Loss', linestyle=':', color=color)

        plt.plot(epochs, val_recon_kl, label='Validation Loss')
        # Get color of the above line
        color = plt.gca().lines[-1].get_color()
        plt.plot(epochs, val_recon, label='Validation Recon Loss', linestyle='--', color=color)
        plt.plot(epochs, val_kl, label='Validation KL Loss', linestyle=':', color=color)

    else:
        train_losses = [loss for loss in train_losses]
        val_losses = [loss for loss in val_losses]
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_latent_grid(model, x_min, x_max, y_min, y_max, n=10, image_size=8):
    """
    Visualizes the 2D latent space by decoding a grid of latent vectors.

    Args:
        model (nn.Module): Trained model with a `.decode()` method.
        x_min, x_max (float): Range for latent dimension 1.
        y_min, y_max (float): Range for latent dimension 2.
        n (int): Number of steps along each axis (n x n grid).
        image_size (float): Size (in inches) of the full figure (square).
    """
    x_vals = torch.linspace(x_min, x_max, n)
    y_vals = torch.linspace(y_max, y_min, n)  
    grid = torch.stack(torch.meshgrid(x_vals, y_vals, indexing='xy'), dim=-1)  
    grid_flat = grid.reshape(-1, 2)  

    with torch.no_grad():
        decoded = model.decode(grid_flat)  

    decoded = decoded.cpu()

    # Plotting
    _, axes = plt.subplots(n, n, figsize=(image_size, image_size))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            img = decoded[i * n + j].numpy().transpose(1, 2, 0)  
            img = img.clip(0, 1)
            ax.imshow(img)
            ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle("Decoded Grid from 2D Latent Space", fontsize=14)
    plt.show()

def plot_image_points(images, points, max_num=128, transparent_black=0.1):
    N = min(len(points), max_num)
    zoom = 0.8

    if transparent_black is not None:
        alpha = transparent_black
        if len(images.shape) == 1:
            # Grayscale to color
            images = np.stack([images, images, images], axis=-1)
        images = np.concatenate([images, alpha+(1-alpha)*(np.sum(images, axis=3, keepdims=True)>1e-1)], axis=-1)


    plt.scatter(points[:N,0], points[:N,1])
    ax = plt.gca()

    for i in range(N):
        im = OffsetImage(images[i], zoom=zoom, cmap='gray')

        abbox = AnnotationBbox(im, (points[i,0], points[i,1]), xycoords='data', frameon=False)
        ax.add_artist(abbox)

    ax.set_aspect('equal')

def set_all_parameters(model, value=1):
    """
    Sets all parameters in the given model to 1.0.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be set.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)

def round_to_list(tensor, decimals=2):
    return str(torch.round(tensor.to(float), decimals=decimals).tolist())

def csv_to_ndarray(filename):
    with open(filename) as f:
        # Skip header
        f.readline()

        data_list = []
        for line in f:
            data_list.append([int(s) for s in line.split(',')])

    data_numpy = np.array(data_list)

    return data_numpy


def rec_load_tiny_data():
    """ Load tiny training dataset

        Returns tuple of length 3: (num_users, num_items, ratings_train_tiny)
        where num_users and num_items are both ints
        and ratings_train_tiny is an ndarray with one row per rating
        each row has three columns containing integers: [user_index, book_index, rating]
        Indices are zero-indexed.
    """
    ratings_train_tiny = csv_to_ndarray('data/ratings_train_tiny.csv')
    
    num_users = np.max(ratings_train_tiny[:, 0]) + 1
    num_items = np.max(ratings_train_tiny[:, 1]) + 1

    return (num_users, num_items, ratings_train_tiny)


def rec_load_data():
    """ Load training and validation datasets

        Returns tuple of length 4: (num_users, num_items, ratings, book_titles)
        where num_users and num_items are both ints
        and ratings is an ndarray with one row per rating
        each row has three columns containing integers: [user_index, book_index, rating]
        Indices are zero-indexed.
        book_titles is a list containing the book titles
    """
    ratings = csv_to_ndarray('data/ratings_train.csv')
    book_titles = list(pd.read_csv('data/books.csv')['title'])
    
    num_users = np.max(ratings[:, 0]) + 1
    num_items = np.max(ratings[:, 1]) + 1

    ratings[:, 2] = ratings[:, 2] - np.mean(ratings[:, 2])
    
    return (num_users, num_items, ratings, book_titles)

def visualize_embeddings(user_embeddings, book_embeddings,
                         user_indices_to_display, item_indices_to_display, book_titles=None):
    """ Plot the embeddings of the users with the embedding of the given book.

        user_indices: list of integers corresponding to the user indices in the same
            order of user_embeddings
        user_embeddings: (num_users, 2) numpy ndarray containing learned embeddings of users
        book_embeddings: (2,) numpy ndarray which is the embedding of the book
        book_title: string of the title of the book

        Returns: matplotlib figure plotting embeddings
    """
    min_alpha = 0.1
    default_alpha = 0.6

    fontsize = 14
    title = f'Embedded Users and Items'

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    # Add x and y axis lines
    # Z order in the back
    ax.axhline(0, color='black', lw=1, zorder=0)
    ax.axvline(0, color='black', lw=1, zorder=0)

    # Plotting embedded users
    for i in user_indices_to_display:
        emb = user_embeddings[i]
        point_color = 'tab:blue'
        point_size = 20
        line_color = 'tab:blue'
        line_width = 1
        label = f'User {i}'
        line_zorder = 1
        point_zorder = 2
        alpha = 1

        # Vector line
        ax.plot([0, emb[0]], [0, emb[1]], line_color, lw=line_width, alpha=alpha*0.5,
                zorder=line_zorder)

        # Vector point
        ax.scatter(emb[0], emb[1], s=point_size, c=point_color,
                zorder=point_zorder, alpha=alpha*0.5)

        # Offset text in the same direction as the vector
        num_chars = len(label)
        offset_scale = 0.9
        horiz_scale = 1.0 + 0.25*num_chars
        offset_shift = np.array([-num_chars*0.25-.05, -0.375])
        xy_offset = offset_scale * emb/np.linalg.norm(emb)
        xy_offset[0] *= horiz_scale
        xy_offset += offset_shift

        ax.annotate(label, (emb[0], emb[1]),
                    xytext=xy_offset, textcoords='offset fontsize',
                    fontsize=fontsize, color=point_color, alpha=alpha*0.7,
                    zorder=point_zorder)

    # Plotting the book embedding
    for j in item_indices_to_display:

        point_color = 'tab:purple'
        line_color = 'tab:purple'

        if book_titles is not None:
            book_title = book_titles[j]
            label = f'Book {j}: {book_title}'
        else:
            label = f'Book {j}'

        # Vector line
        ax.plot([0, book_embeddings[j, 0]], [0, book_embeddings[j, 1]], line_color, lw=line_width, alpha=alpha*0.7,
                zorder=line_zorder)

        # Vector point
        ax.scatter(book_embeddings[j, 0], book_embeddings[j, 1], s=point_size, c=point_color,
                zorder=point_zorder, alpha=alpha)

        # Offset text in the same direction as the vector
        num_chars = len(label)
        offset_scale = 0.9
        horiz_scale = 1.0 + 0.25*num_chars
        offset_shift = np.array([-num_chars*0.25-.05, -0.375])
        xy_offset = offset_scale * book_embeddings[j]/np.linalg.norm(book_embeddings[j])
        xy_offset[0] *= horiz_scale
        xy_offset += offset_shift

        ax.annotate(label, (book_embeddings[j, 0], book_embeddings[j, 1]),
                    xytext=xy_offset, textcoords='offset fontsize',
                    fontsize=fontsize, color=point_color, alpha=alpha,
                    zorder=point_zorder)

    max_vector_mag = np.max(np.linalg.norm(user_embeddings, axis=1))
    max_vector_mag = np.max((np.max(np.linalg.norm(book_embeddings, axis=1)), max_vector_mag))
    max_vector_mag *= 1.1
    ax.set_xlim(-max_vector_mag, max_vector_mag)
    ax.set_ylim(-max_vector_mag, max_vector_mag)

    ax.set_title(title)

def load_mnist_data():
    """ Load MNIST dataset

        Returns an 30000-by-784 numpy ndarray X_train
    """

    X = np.load('mnist.npy').astype(np.float32)
    X_train = np.reshape(X[:30000][:][:], (30000, 784))
    return X_train


def kmeans_loss(X, C, z):
    """ Compute the K-means loss.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
        z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
            representing the cluster index for the i-th point in X

        Returns mean squared distance from each point to the center for its assigned cluster
    """
    N = X.shape[0]
    loss = 0
    for i in range(N):
        diff = X[i] - C[z[i]]
        loss += diff @ diff / N

    return loss

