


import os.path
import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split, ConcatDataset
# from torchvision.datasets import *
from torchvision.datasets import CIFAR100, Flowers102, Food101, CIFAR10, GTSRB, SVHN
from pathlib import Path 
import ssl, os





# max_iter_map = {'pet': 10000, 'flower': 15000, 'cifar100': 20000, 'eurosat': 4000, 'svhn': 16000, 'cifar10': 10000, 'food': 8000, 'stl10':12000, 'coco2017':26000}
# max_epoch_map = {'pet': 87, 'flower': 79, 'cifar100': 52, 'eurosat': 20, 'svhn': 28, 'cifar10': 26, 'food': 14, 'stl10':257, 'coco2017':100}
    

# project_home = project_root = Path(__file__).resolve()  # .parent.parent
project_home = Path(__file__).resolve().parent.parent  # adjust for your env


    
root_map = {
    'cifar10': '.data/',
    'cifar100': '.data/',
    'subcifar': '.data/',
    'gtsrb': '.data/',
    'subgtsrb': '.data/',
    'svhn': '.data/',
    'flower': '.data/',
    'pubfig': '../dataset/pubfig',
    'eurosat': '.data/eurosat',
    'imagenet': '../imagenet/',
    'imagenet64': '../imagenet64/',
    'lisa': '.data/',
    'mnist': '.data/',
    'mnistm': '.data/',
    'food': '.data/',
    'pet': '.data/',
    'resisc': '.data/NWPU-RESISC45/',
    'voc': '.data/',
    'lingspam': '.data/spam/lingspam',
    'stl10': './data/STL10',
    'coco2017': './data/coco'
}

mean_map = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.4914, 0.4822, 0.4465),
    'subcifar': (0.4914, 0.4822, 0.4465),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'imagenet64': (0.485, 0.456, 0.406),
    'flower': (0.485, 0.456, 0.406),
    'caltech101': (0.485, 0.456, 0.406),
    'stl10': (0.485, 0.456, 0.406),
    'iris': (0.485, 0.456, 0.406),
    'fmnist': (0.2860, 0.2860, 0.2860),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.3403, 0.3121, 0.3214),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.3403, 0.3121, 0.3214),#
    'pubfig': (129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0),
    'lisa': (0.3403, 0.3121, 0.3214),  #(0.4563, 0.4076, 0.3895),
    'eurosat': (0.3442, 0.3802, 0.4077),
    'food': (0.5, 0.5, 0.5),
    'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
    'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}

std_map = {
    'cifar10': (0.2023, 0.1994, 0.201),
    'cifar100': (0.2023, 0.1994, 0.201),
    'subcifar': (0.2023, 0.1994, 0.201),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'imagenet64': (0.229, 0.224, 0.225),
    'flower': (0.229, 0.224, 0.225),
    'caltech101': (0.229, 0.224, 0.225),
    'stl10': (0.229, 0.224, 0.225),
    'iris': (0.229, 0.224, 0.225),
    'fmnist': (0.3530, 0.3530, 0.3530),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.2724, 0.2608, 0.2669),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.2724, 0.2608, 0.2669),#
    'pubfig': (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),  # (1.0, 1.0, 1.0), #
    'lisa': (0.2724, 0.2608, 0.2669),  #(0.2298, 0.2144, 0.2259),
    'eurosat': (0.2036, 0.1366, 0.1148),
    'food': (0.5, 0.5, 0.5),
    'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
    'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}

data_size_map = {
    'cifar10': {'train': 50000, 'test': 10000},  # :contentReference[oaicite:0]{index=0}
    'cifar100': {'train': 50000, 'test': 10000},  # :contentReference[oaicite:1]{index=1}
    'subcifar': {'train': None, 'test': None},  # Subset; sizes depend on specific subset
    'gtsrb': {'train': 39209, 'test': 12630},  # German Traffic Sign Recognition Benchmark
    'subgtsrb': {'train': None, 'test': None},  # Subset; sizes depend on specific subset
    'svhn': {'train': 73257, 'test': 26032},  # Street View House Numbers
    'flower': {'train': 1020, 'test': 6149},  # Oxford 102 Flower Dataset
    'pubfig': {'train': 7725, 'test': None},  # Public Figures Face Database
    'eurosat': {'train': 27000, 'test': 3000},  # EuroSAT Dataset
    'imagenet': {'train': 1281167, 'test': 50000},  # ImageNet Large Scale Visual Recognition Challenge
    'imagenet64': {'train': 1281167, 'test': 50000},  # Downsampled ImageNet
    'lisa': {'train': 13384, 'test': 4442},  # LISA Traffic Sign Dataset
    'mnist': {'train': 60000, 'test': 10000},  # :contentReference[oaicite:2]{index=2}
    'mnistm': {'train': 59001, 'test': 9001},  # MNIST-M Dataset
    'food': {'train': 75750, 'test': 25250},  # Food-101 Dataset
    'pet': {'train': 3680, 'test': 3669},  # Oxford-IIIT Pet Dataset
    'resisc': {'train': 25200, 'test': 10800},  # NWPU-RESISC45 Dataset
    'voc': {'train': 5011, 'test': 4952},  # PASCAL VOC 2012
    'lingspam': {'train': 2412, 'test': 803},  # Ling-Spam Dataset
    'stl10': {'train': 8000, 'test': 5000},  
}

size_map = {
    'cifar10': (32, 32),
    'cifar100': (32, 32),
    'subcifar': (32, 32),  # Assuming same as CIFAR
    'gtsrb': (32, 32),  # German Traffic Sign Recognition Benchmark is often resized
    'subgtsrb': (32, 32),  # Assuming same resizing as GTSRB
    'svhn': (32, 32),  # Street View House Numbers dataset is often resized
    'flower': (224, 224),  # Oxford 102 Flower dataset (commonly resized for CNNs)
    'pubfig': (224, 224),  # Public Figures Face Database (often resized)
    'eurosat': (64, 64),  # EuroSAT images are 64×64 by default
    'imagenet': (224, 224),  # Standard size for ImageNet preprocessing
    'imagenet64': (64, 64),  # Downsampled ImageNet
    'lisa': (32, 32),  # LISA Traffic Sign Dataset is often resized for models
    'mnist': (28, 28),  # Standard MNIST image size
    'mnistm': (28, 28),  # MNIST-M is the same size as MNIST
    'food': (224, 224),  # Food-101 images are often resized for CNNs
    'pet': (224, 224),  # Oxford-IIIT Pet Dataset images are typically resized
    'resisc': (256, 256),  # NWPU-RESISC45 Dataset images are 256×256
    'voc': (224, 224),  # PASCAL VOC images are often resized for models
    'lingspam': None,  # Text dataset, no images
    'stl10': (96, 96),
}

num_class_map = {
    'cifar10': 10,
    'cifar100': 100,
    'subcifar': 2,
    'subgtsrb': 2,
    'svhn': 10,
    'pubfig': 83,
    'gtsrb': 43,
    'eurosat': 10,
    'flower': 102,
    'imagenet': 1000,
    'imagenet64': 1000,
    'mnist': 10,
    'mnistm': 10,
    'food': 101,
    'pet': 37,
    'resisc': 45,
    'voc': 21,
    'lingspam': 2,
    'stl10': 10,
    "none": 0
}





# ---------------------------------------------------------------------
def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )



# ---------------------------------------------------------------------
def CIFAR100Loaders(
    root, *, batch_size=256, num_workers=0, val_split=0.1, seed=42, transform=None
):
    """
    Returns train_dl, val_dl, test_dl for CIFAR-100.
    No official val → val = original test, test = val_split % of train.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    root = project_home / root

    full_train = CIFAR100(root, train=True,  transform=transform, download=True)
    official_test = CIFAR100(root, train=False, transform=transform, download=True)

    # split off a *test* subset from the original training data
    torch.manual_seed(seed)
    test_size = int(len(full_train) * val_split)
    train_size = len(full_train) - test_size
    train_set, test_set = random_split(full_train, [train_size, test_size])

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    val_dl   = _make_loader(official_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl



# ---------------------------------------------------------------------
def FlowerLoaders(
    root, *, batch_size=256, num_workers=2, val_split=0.0, seed=42, transform=None, swap_train_test=True
):
    """
    Flowers-102 already has train/val/test splits (as in the paper).
    We honour them; val_split is ignored.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    root = project_home / root

    train_split = "test" if swap_train_test else "train" 
    test_split = "train" if swap_train_test else "test" 

    train_set = Flowers102(root, split=train_split, transform=transform, download=True)
    val_set   = Flowers102(root, split="val"  , transform=transform, download=True)
    test_set  = Flowers102(root, split=test_split , transform=transform, download=True)

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    val_dl   = _make_loader(val_set  , batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl



# ---------------------------------------------------------------------
def Food101Loaders(
    root, *, batch_size=256, num_workers=2, val_split=0.1, seed=42, transform=None
):
    """
    Food-101 has NO official val ⇒ val = official test, test = slice of train.
    """
    root = project_home / root
    full_train = Food101(root, split="train", transform=transform, download=True)
    official_test = Food101(root, split="test", transform=transform, download=True)

    torch.manual_seed(seed)
    test_size = int(len(full_train) * val_split)
    train_size = len(full_train) - test_size
    train_set, test_set = random_split(full_train, [train_size, test_size])

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    val_dl   = _make_loader(official_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl



# ---------------------------------------------------------------------
def CIFAR10Loaders(
    root,
    *,
    batch_size: int = 256,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    transform=None,
):
    """
    Returns train_dl, val_dl, test_dl for CIFAR-10.

    · No official validation ⇒ `val_dl` = official **test** split  
    · `test_dl` = `val_split` % slice of the original training data
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    root = project_home / root

    full_train     = CIFAR10(root, train=True,  transform=transform, download=True)
    official_test  = CIFAR10(root, train=False, transform=transform, download=True)

    torch.manual_seed(seed)
    test_size  = int(len(full_train) * val_split)
    train_size = len(full_train) - test_size
    train_set, test_set = random_split(full_train, [train_size, test_size])

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    val_dl   = _make_loader(official_test, batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------
def GTSRBLoaders(           # GTSRB (German Traffic-Sign)
    root,
    *,
    batch_size: int = 256,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    transform=None,
):
    """
    Returns train_dl, val_dl, test_dl for GTSRB.

    · Official *test* split becomes validation  
    · `test_dl` = `val_split` % slice of the train split
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    root = project_home / root

    full_train     = GTSRB(root, split="train", transform=transform, download=True)
    official_test  = GTSRB(root, split="test" , transform=transform, download=True)

    torch.manual_seed(seed)
    test_size  = int(len(full_train) * val_split)
    train_size = len(full_train) - test_size
    train_set, test_set = random_split(full_train, [train_size, test_size])

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    val_dl   = _make_loader(official_test, batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl

                   
# ---------------------------------------------------------------------
def SVHNLoaders(        # SVHN (Street-View House Numbers)   
    root,
    *,
    batch_size: int = 256,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    transform=None,
):
    """
    Returns train_dl, val_dl, test_dl for SVHN.

    · Official *test* split becomes validation  
    · `test_dl` = `val_split` % slice of the train split
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    root = project_home / root

    full_train     = SVHN(root, split="train", transform=transform, download=True)
    official_test  = SVHN(root, split="test" , transform=transform, download=True)

    torch.manual_seed(seed)
    test_size  = int(len(full_train) * val_split)
    train_size = len(full_train) - test_size
    train_set, test_set = random_split(full_train, [train_size, test_size])

    train_dl = _make_loader(train_set, batch_size, shuffle=True , num_workers=num_workers)
    val_dl   = _make_loader(official_test, batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = _make_loader(test_set , batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl





loader_map = {
    'cifar100': CIFAR100Loaders,
    'flower': FlowerLoaders,
    'food': Food101Loaders,
    'cifar10': CIFAR10Loaders,
    'gtsrb': GTSRBLoaders,
    'svhn': SVHNLoaders,
}




########################################################### old codes #############################################################################
"""
# ToDo
def AllDataSetLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None, val_split=0.0, seed=42):
    import torchvision.datasets
    if shuffle is None: 
        shuffle = (split == 'train')
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)

    if split == 'train' and val_split > 0:
        torch.manual_seed(seed)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        out_dataset = train_dataset
    else:
        out_dataset = dataset

    return DataLoader(
        dataset=out_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )




def CIFAR100Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None, val_split=0.0, seed=42):
    import ssl
    
    root = f"{project_home}/{root}"

    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    # print(os.path.join(os.getcwd(), root))
    dataset = CIFAR100(root, split, transform, download=True)
    
    if split and val_split > 0:       # split = True if split == 'train'
        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        # Split into train and validation subsets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        out_dataset = train_dataset        
    else:    
        out_dataset = dataset
        
    return DataLoader(
        dataset=out_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )



def FlowerLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None, val_split=0.0, seed=42):
    import ssl
    
    root = f"{project_home}/{root}"

    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split_reverse = 'train' if split == 'test' else 'test'
    # dataset = Flowers102(root, split_reverse, transform, download=True)
    # split_ = 'train' if split == 'train' else 'test'
    dataset = Flowers102(root, split_reverse, transform, download=True)
    
    if split == 'train' and val_split > 0:
        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        # Split into train and validation subsets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        out_dataset = train_dataset
    else:
        out_dataset = dataset
        
    return DataLoader(
        dataset=out_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )



def Food101Loader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None, val_split=0.0, seed=42):
    from torchvision.datasets import Food101
    
    root = f"{project_home}/{root}"

    if shuffle is None: 
        shuffle = (split == 'train')  # Shuffle only for training split
    dataset = Food101(root, transform=transform, download=True)

    if split == 'train' and val_split > 0:
        # Split training data into train and validation sets
        torch.manual_seed(seed)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        out_dataset = train_dataset  # Use the training subset
    else:
        out_dataset = dataset

    return DataLoader(
        dataset=out_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

"""
     

