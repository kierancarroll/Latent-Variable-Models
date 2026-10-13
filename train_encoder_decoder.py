import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import SpriteTensorDataset, plot_image_points


def train_encoder_decoder_model(
    model,
    loss_fn,
    train_dataset=SpriteTensorDataset(split="train"),
    val_dataset=SpriteTensorDataset(split="validation"),
    lr=1e-3,
    epochs=25,
    batch_size=128,
    device=None,
    show_embedded_images=True
):
    """
    Training loop for an encoder+decoder PyTorch model.

    Args:
        model (nn.Module): The model to train.
        loss_fn (callable): Loss function. Should take (outputs, targets) and return a scalar loss.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        lr (float): Learning rate.
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training and validation.
        device (str or torch.device): Device to use ("cuda", "cpu", or torch.device object). If None, auto-detect.

    Returns:
        Tuple[List[float], List[float]]: Training and validation losses (averaged per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    first_training_batch = None
    if model.get_latent_dim() > 2:
        show_embedded_images = False

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = None
        total_val_loss = None

        for train_inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            train_inputs = train_inputs.to(device)

            if show_embedded_images and first_training_batch is None:
                first_training_batch = train_inputs
                z = model.encode(train_inputs)
                # If z isn't a tensor, get the first element of the tuple
                if not isinstance(z, torch.Tensor):
                    z = z[0]
                z_np = z.cpu().detach().numpy()
                images_in = train_inputs.cpu().detach().numpy()
                images_in = np.transpose(images_in, (0, 2, 3, 1))
                plot_image_points(images_in, z_np)
                plt.show()

            outputs = model(train_inputs)
            loss = loss_fn(outputs, train_inputs)

            optimizer.zero_grad()
            # If loss_fn returns a tuple, get the first element
            if isinstance(loss, tuple):
                loss[0].backward()
                if total_train_loss is None:
                    total_train_loss = [x.item() for x in loss]
                else:
                    total_train_loss = [x + y.item() for x, y in zip(total_train_loss, loss)]
            else:
                loss.backward()
                if total_train_loss is None:
                    total_train_loss = loss.item()
                else:
                    total_train_loss += loss.item()

            optimizer.step()

        if isinstance(total_train_loss, list):
            avg_train_loss = [x / len(train_loader.dataset) for x in total_train_loss]
        else:
            avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = None
        with torch.no_grad():
            for val_inputs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                val_inputs = val_inputs.to(device)

                outputs = model(val_inputs)

                loss = loss_fn(outputs, val_inputs)
                # If loss_fn returns a tuple, get the first element
                if isinstance(loss, tuple):
                    if total_val_loss is None:
                        total_val_loss = [x.item() for x in loss]
                    else:
                        total_val_loss = [x + y.item() for x, y in zip(total_val_loss, loss)]
                else:
                    if total_val_loss is None:
                        total_val_loss = loss.item()
                    else:
                        total_val_loss += loss.item()


        if isinstance(total_val_loss, list):
            avg_val_loss = [x / len(val_loader.dataset) for x in total_val_loss]
        else:
            avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}")

        if show_embedded_images:
            train_inputs = first_training_batch
            z = model.encode(train_inputs)
            # If z isn't a tensor, get the first element of the tuple
            if not isinstance(z, torch.Tensor):
                z = z[0]

            z_np = z.cpu().detach().numpy()
            images_in = train_inputs.cpu().detach().numpy()
            images_in = np.transpose(images_in, (0, 2, 3, 1))
            plot_image_points(images_in, z_np)
            plt.show()

    return train_losses, val_losses