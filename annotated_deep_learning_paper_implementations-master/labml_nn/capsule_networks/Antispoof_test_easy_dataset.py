from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from labml.configs import BaseConfigs, aggregate, option
from os.path  import join

IMG_SIZE = 100
# IMG_SIZE = 28
# output_chnl =1
def _dataset(is_train, path, transform):
    """
    common dataset transform rule
    is_train - for future experiments with combination of datasets for train/evaluation
    """
    return datasets.ImageFolder(path,
                          # train=is_train,
                          # download=True,
                          transform=transform)
class Antispoof_Easy_Configs(BaseConfigs):
    """
    Configurable Antispoof data set.
    Arguments:
        dataset_name (str): name of the data set, ``Antispoof``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.MNIST): training dataset
        valid_dataset (torchvision.datasets.MNIST): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size
        test_batch_size (int): test batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
        test_loader_shuffle (bool): whether to shuffle test data
    """

    dataset_name: str = 'Antispoof'
    dataset_transforms: transforms.Compose

    input_path = '/home/evgeniy/audio_datasets/Dataset/detection_dataset'
    training_images_filepath = join(input_path, 'train_easy')
    validation_images_filepath = join(input_path, 'val_easy')
    test_images_filepath = join(input_path, 'test_easy')

    train_dataset: datasets.ImageFolder
    valid_dataset: datasets.ImageFolder
    test_dataset: datasets.ImageFolder

    train_loader: DataLoader
    valid_loader: DataLoader

    test_loader: DataLoader

    train_batch_size: int = 8
    valid_batch_size: int = 8
    test_batch_size: int = 8
    # train_batch_size: int = 64               # default value for MNIST results reproducing
    # valid_batch_size: int = 1024
    # test_batch_size: int = 1024
    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False
    test_loader_shuffle: bool = False


@option(Antispoof_Easy_Configs.dataset_transforms)
def Antispoof_transforms():
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=output_chnl),  # decrease accuracy, but could optimize memory a little bit
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # recomendation from MNIST
        transforms.Normalize(mean=[0.5140, 0.4286, 0.3857], std=[0.2468, 0.2237, 0.2181])   # based on antispoof dataset
    ])

@option(Antispoof_Easy_Configs.train_dataset)
def Antispoof_train_dataset(c: Antispoof_Easy_Configs):
    return _dataset(True,c.training_images_filepath, c.dataset_transforms)

@option(Antispoof_Easy_Configs.valid_dataset)
def Antispoof_valid_dataset(c: Antispoof_Easy_Configs):
    return _dataset(True,c.validation_images_filepath, c.dataset_transforms)

@option(Antispoof_Easy_Configs.test_dataset)
def Antispoof_test_dataset(c: Antispoof_Easy_Configs):
    return _dataset(True,c.test_images_filepath, c.dataset_transforms)


@option(Antispoof_Easy_Configs.train_loader)
def Antispoof_train_loader(c: Antispoof_Easy_Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@option(Antispoof_Easy_Configs.valid_loader)
def Antispoof_valid_loader(c: Antispoof_Easy_Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)

@option(Antispoof_Easy_Configs.test_loader)
def Antispoof_test_loader(c: Antispoof_Easy_Configs):
    return DataLoader(c.test_dataset,
                      batch_size=c.test_batch_size,
                      shuffle=c.test_loader_shuffle)

aggregate(Antispoof_Easy_Configs.dataset_name, 'Antispoof',
          (Antispoof_Easy_Configs.dataset_transforms, 'Antispoof_transforms'),
          (Antispoof_Easy_Configs.train_dataset, 'Antispoof_train_dataset'),
          (Antispoof_Easy_Configs.valid_dataset, 'Antispoof_valid_dataset'),
          (Antispoof_Easy_Configs.test_dataset, 'Antispoof_test_dataset'),
          (Antispoof_Easy_Configs.train_loader, 'Antispoof_train_loader'),
          (Antispoof_Easy_Configs.valid_loader, 'Antispoof_valid_loader'),
          (Antispoof_Easy_Configs.test_loader, 'Antispoof_test_loader'))

