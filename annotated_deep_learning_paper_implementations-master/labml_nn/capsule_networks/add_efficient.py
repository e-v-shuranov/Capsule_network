"""
Capsule network for Antispoofing based on solution for MNIST
Lets try to use Efficient based  baselines (checking, compare, fusion etc) )


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

from labml import experiment, tracker
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

from architectures import fornet, weights
from torch.utils.model_zoo import load_url
from isplutils import utils
from blazeface import FaceExtractor, BlazeFace

# device = device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:1" if torch.cuda.is_available() else "cpu"

class Eff_Model(Module):
    """
    ## Model for classifying MNIST digits
    """

    def __init__(self):
        super().__init__()
        # input [batch_size, 256, IMG_SIZE, IMG_SIZE]
        # output         return caps, reconstructions, pred

        """
        Choose an architecture between
        - EfficientNetB4
        - EfficientNetB4ST
        - EfficientNetAutoAttB4
        - EfficientNetAutoAttB4ST
        - Xception
        """
        net_model = 'EfficientNetAutoAttB4'
        """
        Choose a training dataset between
        - DFDC
        - FFPP
        """
        train_db = 'DFDC'
        face_policy = 'scale'
        face_size = 224
        # face_size = 100

        import ssl

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context

        # # from efficientnet_pytorch import EfficientNet
        # classifier = fornet.EfficientNetAutoAttB4().to(device)
        # state_dict = torch.load("/home/evgeniy/models/efficientnet/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth")
        # # state_dict = torch.load("/home/evgeniy/models/efficientnet/efficientnet-b4-6ed6700e.pth")
        # classifier.load_state_dict(state_dict, strict=False)

        model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
        self.classifier = getattr(fornet, net_model)().eval().to(device)
        self.classifier.load_state_dict(load_url(model_url, map_location=device, check_hash=True))


    def forward(self, data: torch.Tensor):

        x = self.classifier(data)

        return x, x, x


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

class Configs(AntispoofConfigs, SimpleTrainValidConfigs):
# class Configs(MNISTConfigs, SimpleTrainValidConfigs):
    """
    Configurations with MNIST data and Train & Validation setup
    """
    epochs: int = 1
    model: nn.Module = 'cnn_network_model'
    MSE_loss = nn.MSELoss()
    antispoof_loss = nn.CrossEntropyLoss()
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
        # data, target = batch[0].to(self.device), batch[1].to(self.device)
        data, target = batch[0].to(device), (1-batch[1]).type(torch.float32).to(device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Whether to log activations
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Run the model
            caps, reconstructions, pred = self.model(data)

        # Calculate the total loss
        # loss = self.margin_loss(caps, target) + 0.0005 * self.reconstruction_loss(reconstructions, data)

        loss = self.MSE_loss(pred, target)
        # loss = self.antispoof_loss(pred, target) + 0.0005 * self.reconstruction_loss(reconstructions, data[:,0:1,:,:])

        tracker.add("loss.", loss)

        # Call accuracy metric
        self.accuracy(pred, target)
        # self.accuracy(pred.argmax(1), target)

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
    return MNISTCapsuleNetworkModel().to(device)

@option(Configs.model)
def cnn_network_model(c: Configs):
    """Set the model"""
    return Eff_Model().to(device)

def data_loading():
    x_tr = x_val = y_tr =  y_val = 0
    return x_tr, x_val, y_tr, y_val
#
# class DuckDuckGooseDataset(Dataset):
#     def __init__(self, x, y=None, device="cpu"):
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         # добавить сюда prepare_shape()
#         if self.y is not None:
#             return \
#                 torch.tensor(self.x[idx], device=device),\
#                 torch.tensor(self.y[idx], device=device)
#
#         return torch.tensor(self.x[idx], device=device)

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

    experiment.create(name='eff_network')
    # experiment.load(run_uuid="d3cb7cf0e9eb11ee8d633cecef777664")
    conf = Configs()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-20})

    experiment.add_pytorch_models({'model': conf.model})

    with experiment.start():
        conf.run()

    experiment.save_checkpoint()

def cnntest():

    """
    Choose an architecture between
    - EfficientNetB4
    - EfficientNetB4ST
    - EfficientNetAutoAttB4
    - EfficientNetAutoAttB4ST
    - Xception
    """
    net_model = 'EfficientNetAutoAttB4'
    """
    Choose a training dataset between
    - DFDC
    - FFPP
    """
    train_db = 'DFDC'
    face_policy = 'scale'
    face_size = 224
    # face_size = 100

    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context



    # # from efficientnet_pytorch import EfficientNet
    # classifier = fornet.EfficientNetAutoAttB4().to(device)
    # state_dict = torch.load("/home/evgeniy/models/efficientnet/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth")
    # # state_dict = torch.load("/home/evgeniy/models/efficientnet/efficientnet-b4-6ed6700e.pth")
    # classifier.load_state_dict(state_dict, strict=False)


    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    classifier = getattr(fornet, net_model)().eval().to(device)
    classifier.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

    # state_dict = torch.load("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/logs/eff_network/3f6cd4daea2411eea3883cecef777664/checkpoints/24480/model.pth")
    # state_dict = torch.load("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/logs/eff_network/6459f07eea2a11ee84863cecef777664/checkpoints/4/model.pth")
    #
    # classifier.load_state_dict(state_dict, strict=True)

    transf = utils.get_transformer(face_policy, face_size, classifier.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/labml_nn/capsule_networks/blazeface/blazeface.pth")
    facedet.load_anchors("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/labml_nn/capsule_networks/blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)


    experiment.create(name='capsule_network_mnist')
    conf = Configs()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-3})

    experiment.add_pytorch_models({'model': conf.model})
    ds = conf.train_dataset


    from PIL import Image

    im_real = Image.open('/home/evgeniy/audio_datasets/Dataset/detection_dataset/example/real/lynaeydofd_fr0.jpg')
    im_fake = Image.open('/home/evgeniy/audio_datasets/Dataset/detection_dataset/example/fake/mqzvfufzoq_fr0.jpg')
    im_real_faces = face_extractor.process_image(img=im_real)
    im_fake_faces = face_extractor.process_image(img=im_fake)

    im_real_face = im_real_faces['faces'][0]  # take the face with the highest confidence score found by BlazeFace
    im_fake_face = im_fake_faces['faces'][0]  # take the face with the highest confidence score found by BlazeFace
    faces_t = torch.stack([transf(image=im)['image'] for im in [im_real_face, im_fake_face]])
    with torch.no_grad():
        faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()
    print('Score for REAL face: {:.4f}'.format(faces_pred[0]))
    print('Score for FAKE face: {:.4f}'.format(faces_pred[1]))


    a = []
    b = []
    result_pred = []
    with torch.no_grad():
        cc = 0
        for i in range(0, len(ds)):
            I = ds[i][0].permute(1, 2, 0).numpy()
            I *= 255  # or any coefficient
            I = I.astype(np.uint8)
            a.append(I)
            b.append(ds.imgs[i][0])
            cc = cc + 1
            if i%100 == 0:
                # faces_t = torch.stack(a)
                faces_t = torch.stack([transf(image=im)['image'] for im in a])
                faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()
                result_pred.append(faces_pred)
                print("faces_pred:", faces_pred)
                a = []
                cc = 0
        if cc>0:
            faces_t = torch.stack([transf(image=im)['image'] for im in a])
            faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()
            result_pred.append(faces_pred)
            print("cc: ",cc, "faces_pred:", faces_pred)

    result_pred = np.concatenate(result_pred)
    labels = ds.targets
    img_files = b







    #
    # with torch.no_grad():
    #     faces_pred = torch.sigmoid(classifier([images[0,0],images[1,0]])).cpu().numpy().flatten()

    print("labels:",labels)
    print('Score for REAL face: ', result_pred)
    np.savetxt('/home/evgeniy/models/efficientnet/train_res.csv', [p for p in zip(labels, result_pred, img_files)], delimiter=';', fmt='%s')
#Huggingface format
# model.save_pretrained("./your_file_name")
# BertModel.from_pretrained("./your_file_name")


if __name__ == '__main__':
    cnntest()
    # main()
