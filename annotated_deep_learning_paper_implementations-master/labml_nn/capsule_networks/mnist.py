"""
Capsule network for Antispoofing based on solution for MNIST


Comments from MNIST:
---
title: Classify MNIST digits with Capsule Networks
summary: Code for training Capsule Networks on MNIST dataset
---

# Classify MNIST digits with Capsule Networks

This is an annotated PyTorch code to classify MNIST digits with PyTorch.

This paper implements the experiment described in paper
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).
"""
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml import lab, experiment, tracker
from labml.configs import option
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.metrics.accuracy import AccuracyDirect
from labml_helpers.module import Module
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_nn.capsule_networks import Squash, Router, MarginLoss

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

# %matplotlib inline
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# from imutils import paths

# from Antispoofdataset import AntispoofConfigs, IMG_SIZE
from Antispoof_test_dataset import AntispoofConfigs, IMG_SIZE

device = device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MNISTCapsuleNetworkModel(Module):
    """
    ## Model for classifying MNIST digits
    """

    def __init__(self):
        super().__init__()
        # First convolution layer has $256$, $9 \times 9$ convolution kernels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        # The second layer (Primary Capsules) s a convolutional capsule layer with $32$ channels
        # of convolutional $8D$ capsules ($8$ features per capsule).
        # That is, each primary capsule contains 8 convolutional units with a 9 × 9 kernel and a stride of 2.
        # In order to implement this we create a convolutional layer with $32 \times 8$ channels and
        # reshape and permutate its output to get the capsules of $8$ features each.
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2, padding=0)
        self.squash = Squash()

        # Routing layer gets the $32 \times 6 \times 6$ primary capsules and produces $10$ capsules.
        # Each of the primary capsules have $8$ features, while output capsules (Digit Capsules)
        # have $16$ features.
        # The routing algorithm iterates $3$ times.
        # self.digit_capsules = Router(32 * 6 * 6, 10, 8, 16, 3)
        out_size = int((IMG_SIZE - 16) / 2)
        self.digit_capsules = Router(32 * out_size * out_size, 10, 8, 16, 3)

        # This is the decoder mentioned in the paper.
        # It takes the outputs of the $10$ digit capsules, each with $16$ features to reproduce the
        # image. It goes through linear layers of sizes $512$ and $1024$ with $ReLU$ activations.
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 784),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE),
            nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor):
        """
        `data` are the MNIST images, with shape `[batch_size, 1, 28, 28]`
        """
        # print("data.shape:",data.shape)
        # print("data[0]",data[0])

        # Pass through the first convolution layer.
        # Output of this layer has shape `[batch_size, 256, 20, 20]`
        x = F.relu(self.conv1(data))
        # Pass through the second convolution layer.
        # Output of this has shape `[batch_size, 32 * 8, 6, 6]`.
        # *Note that this layer has a stride length of $2$*.
        x = self.conv2(x)

        # Resize and permutate to get the capsules
        # caps = x.view(x.shape[0], 8, 32 * 6 * 6).permute(0, 2, 1)
        # caps = x.view(x.shape[0], 8, 32 * 292 * 292).permute(0, 2, 1)
        num_of_capsuls = int(x.shape[2]*x.shape[3]*x.shape[1]/8)
        caps = x.view(x.shape[0], 8, num_of_capsuls).permute(0, 2, 1)
        # Squash the capsules
        caps = self.squash(caps)
        # Take them through the router to get digit capsules.
        # This has shape `[batch_size, 10, 16]`.
        caps = self.digit_capsules(caps)

        # Get masks for reconstructioon
        with torch.no_grad():
            # The prediction by the capsule network is the capsule with longest length
            pred = (caps ** 2).sum(-1).argmax(-1)
            # Create a mask to maskout all the other capsules
            mask = torch.eye(10, device=data.device)[pred]

        # Mask the digit capsules to get only the capsule that made the prediction and
        # take it through decoder to get reconstruction
        reconstructions = self.decoder((caps * mask[:, :, None]).view(x.shape[0], -1))
        # Reshape the reconstruction to match the image dimensions
        # reconstructions = reconstructions.view(-1, 1, 28, 28)
        reconstructions = reconstructions.view(-1, 1, IMG_SIZE, IMG_SIZE)

        return caps, reconstructions, pred
class Configs_test(AntispoofConfigs, SimpleTrainValidConfigs):
# class Configs(MNISTConfigs, SimpleTrainValidConfigs):
    """
    Configurations with MNIST data and Train & Validation setup
    """
    epochs: int = 1
    model: nn.Module = 'capsule_network_model'
    # reconstruction_loss = nn.MSELoss()
    # margin_loss = MarginLoss(n_labels=10)
    accuracy = AccuracyDirect()
    def init(self):
        # Print losses and accuracy to screen
        tracker.set_scalar('loss.*', True)
        tracker.set_scalar('accuracy.*', True)
        # We need to set the metrics to calculate them for the epoch for training and validation
        self.state_modules = [self.accuracy]
    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method gets called by the trainer
        """
        # Get the images and labels and move them to the model's device
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        # Whether to log activations
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Run the model
            caps, reconstructions, pred = self.model(data)
        # Call accuracy metric
        self.accuracy(pred, target)


class Configs(AntispoofConfigs, SimpleTrainValidConfigs):
# class Configs(MNISTConfigs, SimpleTrainValidConfigs):
    """
    Configurations with MNIST data and Train & Validation setup
    """
    epochs: int = 10
    model: nn.Module = 'capsule_network_model'
    reconstruction_loss = nn.MSELoss()
    margin_loss = MarginLoss(n_labels=10)
    accuracy = AccuracyDirect()

    def init(self):
        # Print losses and accuracy to screen
        tracker.set_scalar('loss.*', True)
        tracker.set_scalar('accuracy.*', True)

        # We need to set the metrics to calculate them for the epoch for training and validation
        self.state_modules = [self.accuracy]

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method gets called by the trainer
        """
        # Set the model mode
        self.model.train(self.mode.is_train)

        # Get the images and labels and move them to the model's device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Whether to log activations
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Run the model
            caps, reconstructions, pred = self.model(data)

        # Calculate the total loss
        loss = self.margin_loss(caps, target) + 0.0005 * self.reconstruction_loss(reconstructions, data[:,0:1,:,:])
        # loss = self.margin_loss(caps, target)
        # loss = self.reconstruction_loss(reconstructions, data[:,0:1,:,:])
        tracker.add("loss.", loss)

        # Call accuracy metric
        self.accuracy(pred, target)

        if self.mode.is_train:
            loss.backward()

            self.optimizer.step()
            # Log parameters and gradients
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()

            tracker.save()


@option(Configs.model)
def capsule_network_model(c: Configs):
    """Set the model"""
    return MNISTCapsuleNetworkModel().to(c.device)

def data_loading():
    x_tr = x_val = y_tr =  y_val = 0
    return x_tr, x_val, y_tr, y_val

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1

    plt.show()

def main():
    """
    Run the experiment
    """
    # input_path = '/home/evgeniy/audio_datasets/Dataset/detection_dataset'
    # training_images_filepath = join(input_path, 'train')
    # val_images_filepath = join(input_path, 'val')
    # test_images_filepath = join(input_path, 'test')
    # from torchvision import datasets, transforms
    #
    # transform = transforms.Compose([
    #     # you can add other transformations in this list
    #     transforms.Resize(IMG_SIZE),
    #     transforms.ToTensor()
    # ])
    #
    # train_dataset_img = datasets.ImageFolder(training_images_filepath, transform = transform )
    # val_dataset_img = datasets.ImageFolder(val_images_filepath, transform = transform )
    # test_dataset_img = datasets.ImageFolder(test_images_filepath, transform = transform )
    #
    # trainloader = torch.utils.data.DataLoader(train_dataset_img, batch_size=32)
    # valloader = torch.utils.data.DataLoader(val_dataset_img, batch_size=32)
    # testloader = torch.utils.data.DataLoader(test_dataset_img, batch_size=32)
    # from torchvision import transforms
    #
    # mean = 0.0
    # std = 0.0
    # for img, _ in train_dataset_img:
    #     # mean += img.sum([1,2])/torch.numel(img[0])
    #     mean += img.mean([1, 2])
    #     std += img.std([1, 2])
    # for img, _ in val_dataset_img:
    #     # mean += img.sum([1,2])/torch.numel(img[0])
    #     mean += img.mean([1, 2])
    #     std += img.std([1, 2])
    # for img, _ in test_dataset_img:
    #     # mean += img.sum([1,2])/torch.numel(img[0])
    #     mean += img.mean([1, 2])
    #     std += img.std([1, 2])
    #
    #
    # mean = mean / (len(train_dataset_img) + len(val_dataset_img) + len(test_dataset_img))
    # std = std / (len(train_dataset_img) + len(val_dataset_img) + len(test_dataset_img))
    # print(mean)
    # print(std)
    #
    # return

# check images
#     images_2_show = []
#     titles_2_show = []
#     data_iter = iter(trainloader)
#     for i in range(0, 10):
#
#         images, labels = next(data_iter)
#         r = random.randint(0, 32)
#         images_2_show.append(images[0,0])
#         titles_2_show.append('training image [' + str(r) + '] = ' + str(labels))
#
#     data_iter = iter(valloader)
#     for i in range(0, 5):
#
#         images, labels = next(data_iter)
#         r = random.randint(0, 32)
#         images_2_show.append(images[0,0])
#         titles_2_show.append('test image [' + str(r) + '] = ' + str(labels))
#
#     show_images(images_2_show, titles_2_show)
# check images------------------------------------



    # x_tr, x_val, y_tr, y_val = data_loading()
    # train_dset = DuckDuckGooseDataset(x_tr, y_tr, device=device)
    # val_dset = DuckDuckGooseDataset(x_val, y_val, device=device)
    # num_workers = 0
    #
    # train_loader = DataLoader(train_dset, batch_size=41, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dset, batch_size=41, shuffle=False, num_workers=num_workers)

    # eval_only = True
    # if eval_only:
    #     experiment.load(run_uuid = "c78234c2e9c111eeb0863cecef777664")
    #     # experiment.create(uuid = "c78234c2e9c111eeb0863cecef777664")
    #     experiment.evaluate()
    #     return
    # lab.experiments_path = "/home/evgeniy/models/efficientnet/ckpt"

    experiment.create(name='capsule_network_mnist')
    # experiment.load(run_uuid="d3cb7cf0e9eb11ee8d633cecef777664")
    # eval_only = True
    # if eval_only:
    #     experiment.evaluate()
    #     return

    # experiment.create(name='capsule_network_mnist')
    conf = Configs()

    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-3})
    experiment.add_pytorch_models({'model': conf.model})

    with experiment.start():
        conf.run()

    experiment.save_checkpoint()
    print("QQQQQ")


def main_MNIST():
    """
    Run the experiment
    """
    experiment.create(name='capsule_network_mnist')
    conf = Configs()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-3})

    experiment.add_pytorch_models({'model': conf.model})

    with experiment.start():
        conf.run()

def test_capsul():
    experiment.create(name='capsule_network_mnist')
    experiment.load(run_uuid="d1f2635ae9e911ee9aac3cecef777664")
    conf = Configs_test()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-20})
    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()

if __name__ == '__main__':
    main()
    # main_MNIST()
    # test_capsul()
