from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from labml import lab
from labml.configs import BaseConfigs, aggregate, option
from os.path  import join

IMG_SIZE = 100


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def _dataset(is_train, path, transform):
    return datasets.ImageFolder(path,
                          # train=is_train,
                          # download=True,
                          transform=transform)

def _train_test_dataset(is_train, path,path2, transform):
    ds = datasets.ImageFolder(path,
                          # train=is_train,
                          # download=True,
                          transform=transform)
    test_transf =  transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # recomendation from MNIST example
        transforms.Normalize(mean=[0.5140, 0.4286, 0.3857], std=[0.2468, 0.2237, 0.2181])   # based on antispoof dataset
    ])

    test = datasets.ImageFolder(path2,
                          # train=is_train,
                          # download=True,
                          transform=test_transf)
    train_with_test_sets = torch.utils.data.ConcatDataset([ds, test])
    return train_with_test_sets


def _train_aug_dataset(is_train, path, transform):
    ds = datasets.ImageFolder(path,
                          # train=is_train,
                          # download=True,
                          transform=transform)

    random = torch.rand(1).numpy()*3
    aug_transf =  transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomResizedCrop(size=(int(IMG_SIZE - random), int(IMG_SIZE - random)), antialias=True),
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5140, 0.4286, 0.3857], std=[0.2468, 0.2237, 0.2181]),  # based on antispoof dataset
        # AddGaussianNoise(0., 1.)
    ])

    aug = datasets.ImageFolder(path,
                          # train=is_train,
                          # download=True,
                          transform=aug_transf)
    train_with_aug_sets = torch.utils.data.ConcatDataset([ds, aug])
    return train_with_aug_sets




class AntispoofConfigs(BaseConfigs):
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

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """

    dataset_name: str = 'Antispoof'
    dataset_transforms: transforms.Compose

    input_path = '/home/evgeniy/audio_datasets/Dataset/detection_dataset'
    # training_images_filepath = join(input_path, 'example')
    training_images_filepath = join(input_path, 'train_easy')
    validation_images_filepath = join(input_path, 'val_easy')
    test_images_filepath = join(input_path, 'test_easy')

    train_dataset: datasets.ImageFolder
    valid_dataset: datasets.ImageFolder
    test_dataset: datasets.ImageFolder

    train_loader: DataLoader
    valid_loader: DataLoader

    test_loader: DataLoader

    train_batch_size: int = 32
    valid_batch_size: int = 32
    test_batch_size: int = 32
    # train_batch_size: int = 64
    # valid_batch_size: int = 1024
    # test_batch_size: int = 1024
    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False
    test_loader_shuffle: bool = False


@option(AntispoofConfigs.dataset_transforms)
def Antispoof_transforms():
    # return transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # recomendation from MNIST example
        transforms.Normalize(mean=[0.5140, 0.4286, 0.3857], std=[0.2468, 0.2237, 0.2181])   # based on antispoof dataset

    ])


@option(AntispoofConfigs.train_dataset)
def Antispoof_train_dataset(c: AntispoofConfigs):
    return _train_test_dataset(True,c.training_images_filepath, c.test_images_filepath, c.dataset_transforms)


@option(AntispoofConfigs.valid_dataset)
def Antispoof_valid_dataset(c: AntispoofConfigs):
    return _dataset(True,c.validation_images_filepath, c.dataset_transforms)
    # return _dataset(True,c.test_images_filepath, c.dataset_transforms)   # to replase valid -> test

@option(AntispoofConfigs.test_dataset)
def Antispoof_test_dataset(c: AntispoofConfigs):
    return _dataset(True,c.test_images_filepath, c.dataset_transforms)


@option(AntispoofConfigs.train_loader)
def Antispoof_train_loader(c: AntispoofConfigs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@option(AntispoofConfigs.valid_loader)
def Antispoof_valid_loader(c: AntispoofConfigs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)

@option(AntispoofConfigs.test_loader)
def Antispoof_test_loader(c: AntispoofConfigs):
    return DataLoader(c.test_dataset,
                      batch_size=c.test_batch_size,
                      shuffle=c.test_loader_shuffle)

aggregate(AntispoofConfigs.dataset_name, 'Antispoof',
          (AntispoofConfigs.dataset_transforms, 'Antispoof_transforms'),
          (AntispoofConfigs.train_dataset, 'Antispoof_train_dataset'),
          (AntispoofConfigs.valid_dataset, 'Antispoof_valid_dataset'),
          (AntispoofConfigs.test_dataset, 'Antispoof_test_dataset'),
          (AntispoofConfigs.train_loader, 'Antispoof_train_loader'),
          (AntispoofConfigs.valid_loader, 'Antispoof_valid_loader'),
          (AntispoofConfigs.test_loader, 'Antispoof_test_loader'))

