import matplotlib.pyplot as plt
import numpy as np
import os
import wget
import zipfile
from random import randint
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# Select back-end device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

print(f"Using {device} as default device")

# Dowloading dataset
dataset_zip_path = "utensils.zip"
dataset_src_url = "https://homepages.inf.ed.ac.uk/rbf/UTENSILS/raw.zip"
if not os.path.isfile(dataset_zip_path):
    wget.download(dataset_src_url, dataset_zip_path)

dataset_root_base_path = "dataset/utensils"
if not os.path.isdir(dataset_root_base_path):
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_root_base_path)

dataset_root_path = os.path.join(dataset_root_base_path, "RAW IMAGES")

print(os.listdir(dataset_root_path))

# Calculating normalization mean/std
transforms_tmp = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
our_dataset = ImageFolder(root=dataset_root_path, transform=transforms_tmp)
classes = our_dataset.classes

norm_mean = (0.,)
norm_std = (0.,)
dataset_len = len(our_dataset)
for img, _ in our_dataset:
    img = img.numpy().transpose((1, 2, 0))
    w, h, c = img.shape
    img = np.resize(img, (w * h, 3))
    norm_mean += img.mean(0)
    norm_std += img.std(0)

norm_mean /= dataset_len
norm_std /= dataset_len
print(f"Dataset normalization mean: {norm_mean}, std: {norm_std}")

# Visualizing samples
def sample():
    img, lbl = our_dataset[randint(0, dataset_len - 1)]
    img = img.numpy().transpose((1, 2, 0))
    return (img, lbl)

plt.figure(figsize=(16, 9))
plt.subplot(2, 2, 1)
img, lbl = sample()
plt.imshow(img)
plt.title(classes[lbl])

plt.subplot(2, 2, 2)
img, lbl = sample()
plt.imshow(img)
plt.title(classes[lbl])

plt.subplot(2, 2, 3)
img, lbl = sample()
plt.imshow(img)
plt.title(classes[lbl])

plt.subplot(2, 2, 4)
img, lbl = sample()
plt.imshow(img)
plt.title(classes[lbl])
plt.show()

# Torch dataset
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=(0, 80)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.4, contrast=.4, hue=.2),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.2, 4)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
our_dataset = ImageFolder(root=dataset_root_path, transform=transforms_train)

validation_split = 0.3
n_data = len(our_dataset)
n_validation = int(validation_split * n_data)
n_train = n_data - n_validation

train_set, test_set = random_split(our_dataset, [n_train, n_validation], generator=torch.Generator(device=device))

batch_size = 64
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))

print(f"Training dataset has {len(train_set)} examples")
print(f"Test dataset has {len(test_set)} examples")
print(f"Training dataloader has {len(train_dataloader)} batches")
print(f"Test dataloader has {len(test_dataloader)} batches")
print("Using transforms:")
print(transforms_train)

# Sanity check
iterator = iter(train_dataloader)
images, labels = next(iterator)

plt.figure(figsize=(16, 9))
img_grid = make_grid(images)
img = img_grid.numpy().transpose((1, 2, 0))
img = norm_std * img + norm_mean
img = np.clip(img, 0, 1)
plt.imshow(img)
plt.show()
print([classes[labels[i]] for i in range(batch_size)])