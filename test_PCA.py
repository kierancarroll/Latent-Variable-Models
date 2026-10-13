from torch.utils.data import DataLoader
from utils import display_samples, SpriteTensorDataset
from models.PCA import PCAEncoder

# Fitting PCA with K=2
pca_2 = PCAEncoder(2)

# See how well PCA works on the training set
train_loader = DataLoader(SpriteTensorDataset(split="train"), batch_size=10)
pca_2.fit(train_loader)

X = next(iter(train_loader))
Z = pca_2.encode(X)
X_recon = pca_2.decode(Z)

display_samples(X)
for z in Z:
    print(f"{z[0]:.2f}, {z[1]:.2f}")
display_samples(X_recon)


# Generate new images
samples = [pca_2.sample() for _ in range(10)]
display_samples(samples)