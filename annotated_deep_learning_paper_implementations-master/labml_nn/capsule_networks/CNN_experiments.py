"""

CNN based solucions (simple CNN , Efficientnet etc)  Neural Networks baselines (checking, compare, fusion etc) )

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

from labml import experiment, tracker
from labml.configs import option

from labml_helpers.metrics.accuracy import AccuracyDirect
from labml_helpers.module import Module
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_nn.capsule_networks import Squash, Router, MarginLoss

import numpy as np # linear algebra

import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# from Capsule_experiments import CapsuleNetworkModel, Configs
# from Antispoof_test_augmented_dataset import Antispoof_Aug_Configs, IMG_SIZE
# from Antispoof_test_dataset import AntispoofConfigs, IMG_SIZE
# from Antispoof_test_easy_dataset import Antispoof_Easy_Configs, IMG_SIZE

device = "cuda:1" if torch.cuda.is_available() else "cpu"

class CNNNetworkModel(Module):
    """
    ## Simple CNN model for classifying fake faces dataset
    """

    def __init__(self):
        super().__init__()
        # input [batch_size, 256, IMG_SIZE, IMG_SIZE]
        # output         return caps, reconstructions, pred


        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=5, stride=(2, 2), device=device)
        self.conv2 = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=5, stride=(2, 2), device=device)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=(1, 1), device=device)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=(1, 1), device=device)
        self.conv5 = nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=5, stride=(2, 2), device=device)

        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm2d(num_features=32, device=device)
        self.bn2 = nn.BatchNorm2d(num_features=64, device=device)
        self.bn3 = nn.BatchNorm2d(num_features=96, device=device)
        self.bn4 = nn.BatchNorm2d(num_features=64, device=device)
        self.bn5 = nn.BatchNorm2d(num_features=32, device=device)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(143648, 256)        # for  IMG_SIZE 600
        self.fc2 = nn.Linear(256, 2)

        # self.fc1 = nn.Linear(800, 128)      # for  IMG_SIZE 100
        # self.fc2 = nn.Linear(128, 2)

        #  ---------------additional experiments with adding decoder from Capsule to CNN----------------------

        # This is the decoder mentioned in the paper.
        # It takes the outputs of the $10$ digit capsules, each with $16$ features to reproduce the
        # image. It goes through linear layers of sizes $512$ and $1024$ with $ReLU$ activations.
        # self.decoder = nn.Sequential(
        #     nn.Linear(107584, 512),
        #     # nn.Linear(16 * 10, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     # nn.Linear(1024, 784),
        #     nn.Linear(1024, IMG_SIZE * IMG_SIZE),
        #     nn.Sigmoid()
        # )

    def forward(self, data: torch.Tensor):
        """
        `data` [batch_size, 1, 100, 100]`  or  [batch_size, 1, 600, 600]`
        """
        # print("data.shape:",data.shape)
        # print("data[0]",data[0])

        # Pass through the first convolution layer.
        # Output of this layer has shape `[batch_size, 256, 20, 20]`
        x = F.relu(self.conv1(data))
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout5(x)

        x = self.flat(x)

        pred = self.fc1(x)
        pred = self.fc2(pred)

        # pred = pred.argmax()

        #  ---------------additional experiments with adding decoder from Capsule to CNN----------------------

        # Mask the digit capsules to get only the capsule that made the prediction and
        # take it through decoder to get reconstruction
        # reconstructions = self.decoder(x)
        # # Reshape the reconstruction to match the image dimensions
        # reconstructions = reconstructions.view(-1, 1, IMG_SIZE, IMG_SIZE)

        return pred, pred, pred

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
    Run the experiment with simple CNN
    """

    experiment.create(name='cnn_network_model')
    conf = Configs()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-3})

    experiment.add_pytorch_models({'model': conf.model})

    with experiment.start():
        conf.run()


def cnntest():
    """
        Run embeding extraction from Efficientnet
    """
    from architectures import fornet, weights
    from torch.utils.model_zoo import load_url
    from isplutils import utils
    from blazeface import FaceExtractor, BlazeFace
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



    # #------------ additional experiments with diferent pretrain and fintuned Efficientnet-----------------
    # classifier = fornet.EfficientNetAutoAttB4().to(device)
    # state_dict = torch.load("/home/evgeniy/models/efficientnet/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth")
    # state_dict = torch.load("/home/evgeniy/models/efficientnet/efficientnet-b4-6ed6700e.pth")
    # classifier.load_state_dict(state_dict, strict=False)


    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    classifier = getattr(fornet, net_model)().eval().to(device)
    classifier.load_state_dict(load_url(model_url, map_location=device, check_hash=True))


    transf = utils.get_transformer(face_policy, face_size, classifier.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/labml_nn/capsule_networks/blazeface/blazeface.pth")
    facedet.load_anchors("/home/evgeniy/mytest/annotated_deep_learning_paper_implementations-master/labml_nn/capsule_networks/blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)

    # ---------dataset uploading ---------------------
    conf = Configs()
    ds = conf.test_dataset


    from PIL import Image

    im_real = Image.open('/home/evgeniy/audio_datasets/Dataset/detection_dataset/example/real/lynaeydofd_fr0.jpg')
    im_fake = Image.open('/home/evgeniy/audio_datasets/Dataset/detection_dataset/example/fake/mqzvfufzoq_fr0.jpg')
    im_real_faces = face_extractor.process_image(img=im_real)
    im_fake_faces = face_extractor.process_image(img=im_fake)

    im_real_face = im_real_faces['faces'][0]  # take the face with the highest confidence score found by BlazeFace
    im_fake_face = im_fake_faces['faces'][0]  # take the face with the highest confidence score found by BlazeFace
    faces_t = torch.stack([transf(image=im)['image'] for im in [im_real_face, im_fake_face]])
    with torch.no_grad():
        # faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()   # experiments with Efficient net classifier
        features = (classifier(faces_t.to(device))).cpu().numpy()
    # print('Score for REAL face: {:.4f}'.format(faces_pred[0]))                               # experiments with Efficient net classifier
    # print('Score for FAKE face: {:.4f}'.format(faces_pred[1]))                               # experiments with Efficient net classifier
    print("features: ", features)

    #  --------------  feature extraction ----------------------------------
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
                faces_t = torch.stack([transf(image=im)['image'] for im in a])
                features = (classifier(faces_t.to(device))).cpu().numpy()
                # faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()   # experiments with Efficient net classifier
                result_pred.append(features)
                print("ds features :", features)
                a = []
                cc = 0
        if cc>0:
            faces_t = torch.stack([transf(image=im)['image'] for im in a])
            # faces_pred = torch.sigmoid(classifier(faces_t.to(device))).cpu().numpy().flatten()       # experiments with Efficient net classifier
            features = (classifier(faces_t.to(device))).cpu().numpy()
            result_pred.append(features)
            print("cc: ",cc, "ds features:", features)

    result_pred = np.concatenate(result_pred)
    labels = ds.targets
    img_files = b


    #  --------------------- storage embedings -----------------------------------
    print("labels:",labels)
    print('Score for REAL face: ', result_pred)
    # np.savetxt('/home/evgeniy/models/efficientnet/train_res.csv', [p for p in zip(labels, result_pred, img_files)], delimiter=';', fmt='%s')  # experiments with Efficient net classifier
    list_for_storage = [p for p in zip(labels, result_pred, img_files)]
    import pickle
    feature_train_path='/home/evgeniy/models/efficientnet/train_easy_features.pkl'
    feature_val_path='/home/evgeniy/models/efficientnet/val_easy_features.pkl'
    feature_test_path='/home/evgeniy/models/efficientnet/test_easy_features.pkl'
    with open(feature_test_path, 'wb') as f:
        pickle.dump(list_for_storage, f, pickle.HIGHEST_PROTOCOL)

def ML_antispoof():
    """
    Run ML classifier based on embedings from Efficientnet
    """
    import pickle
    # ---------------Embedings for ALL antispoof dataset  ----------------------------------
    # feature_train_path='/home/evgeniy/models/efficientnet/train_features.pkl'
    # feature_val_path='/home/evgeniy/models/efficientnet/val_features.pkl'
    feature_test_path_hard='/home/evgeniy/models/efficientnet/test_features.pkl'

    # ---------------Embedings only for easy part of antispoof dataset  ----------------------------------
    feature_train_path='/home/evgeniy/models/efficientnet/train_easy_features.pkl'
    feature_val_path='/home/evgeniy/models/efficientnet/val_easy_features.pkl'
    feature_test_path='/home/evgeniy/models/efficientnet/test_easy_features.pkl'

    with open(feature_train_path, 'rb') as f:
        train_features = pickle.load(f)
    with open(feature_val_path, 'rb') as f:
        val_features = pickle.load(f)
    with open(feature_test_path, 'rb') as f:
        test_features = pickle.load(f)
    with open(feature_test_path_hard, 'rb') as f:
        test_features_hard = pickle.load(f)

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA

    for ii in range(10, 100):
        pca = PCA(n_components=ii)

        clf = svm.SVC()
        X = []
        Y = []
        f_train = []
        #  --- use (train and val) for svm + PCA training ----
        for i in range(0,len(val_features)):
            X.append(val_features[i][1].tolist())
            Y.append(val_features[i][0])
            f_train.append(train_features[i][2])

        for i in range(0,len(train_features)):
            X.append(train_features[i][1].tolist())
            Y.append(train_features[i][0])
            f_train.append(train_features[i][2])

        X_test = []
        Y_test = []
        f_test = []

        for i in range(0,len(test_features)):
            X_test.append(test_features[i][1].tolist())
            Y_test.append(test_features[i][0])
            f_test.append(train_features[i][2])

        X_test_hard = []
        Y_test_hard = []
        f_test_hard = []
        for i in range(0,len(test_features_hard)):
            X_test_hard.append(test_features_hard[i][1].tolist())
            Y_test_hard.append(test_features_hard[i][0])
            f_test_hard.append(train_features[i][2])


        pca.fit(X)
        X_pca = pca.transform(X)
        clf.fit(X_pca, Y)

        X_test_pca = pca.transform(X_test)
        y_pred = clf.predict(X_test_pca)
        y_pred_tr = clf.predict(X_pca)

        X_test_pca_hard = pca.transform(X_test_hard)
        y_pred_hard = clf.predict(X_test_pca_hard)


        print(ii, ": acc: ",accuracy_score(Y_test, y_pred),": All data acc: ",accuracy_score(Y_test_hard, y_pred_hard), ": train  acc: ",accuracy_score(Y, y_pred_tr))




def capsule_test():
    """
        Run embeding extraction from Capsule Network
    """
    # from Capsule_experiments import CapsuleNetworkModel, Configs
    from Capsule_classification_experiments import CapsuleNetworkModel, Configs
    # Capsule 2 capsule easy 100 Size checkpoint
    # last_step = "/home/evgeniy/mytest/Capsule_network/annotated_deep_learning_paper_implementations-master/logs/capsule_network_mnist/29111910ef4e11eea8593cecef777664/checkpoints/10650/model.pth"
    # Capsule 2 capsule ALL 600 Size checkpoint from  Capsule_experiments
    # last_step = "/home/evgeniy/mytest/Capsule_network/annotated_deep_learning_paper_implementations-master/logs/capsule_network_mnist/b01484d6ef7d11eebff23cecef777664/checkpoints/16320/model.pth"
    # Capsule 4 capsule ALL 600 Size checkpoint from Capsule_classification_experiments
    last_step = "/home/evgeniy/mytest/Capsule_network/annotated_deep_learning_paper_implementations-master/logs/capsule_classification_network/b6fa3f64ef8411ee99863cecef777664/checkpoints/34560/model.pth"
    classifier = CapsuleNetworkModel()
    classifier.load_state_dict(torch.load(last_step, map_location="cuda:1"), strict=False)
    classifier.to(device)
    # # MNIST pretrained Capsule embedings  0.99 acc on mnist
    # from mnist import MNISTCapsuleNetworkModel
    # # MNIST checkpoint
    # mnist_last_step = "/home/evgeniy/mytest/Capsule_network/annotated_deep_learning_paper_implementations-master/logs/MNIST_capsule_network_mnist/0c9037deef6d11ee98f83cecef777664/checkpoints/300000/model.pth"
    # classifier = MNISTCapsuleNetworkModel()
    # classifier.load_state_dict(torch.load(mnist_last_step), strict=False)

    # ---------dataset uploading ---------------------

    conf = Configs()

    ds = conf.train_dataset

    a = []
    with torch.no_grad():
        im_real = ds[0][0].unsqueeze(0)
        a.append(im_real)
        im_fake = ds[100][0].unsqueeze(0)
        a.append(im_fake)

        b = torch.cat(a)
        b = b.to(device)
        caps, reconstructions, pred = classifier(b)
    #     Classes fake 0 real 1
    # print("caps.shape", caps.shape, "pred", pred, "classes", [ds.targets[0],ds.targets[100]],"names:", ds.imgs[0][0],ds.imgs[100][0])
    print("caps.shape", caps.shape, "pred", pred, "classes", [ds.targets[0],ds.targets[100]])

    #  --------------  feature extraction ----------------------------------
    a = []
    names = []
    result_pred = []
    result_features = []
    with torch.no_grad():
        cc = 0
        for i in range(0, len(ds)):
            I = ds[i][0].unsqueeze(0)
            a.append(I)
            names.append(ds.imgs[i][0])
            # names.append("mnisp")
            cc = cc + 1
            if i%8 == 0:
                b = torch.cat(a)
                b = b.to(device)
                features, reconstructions, pred = classifier(b)
                result_features.append(features.cpu().detach())
                result_pred.append(pred.cpu().detach())
                print("ds features :", features)
                a = []
                cc = 0
        if cc>0:
            b = torch.cat(a)
            b = b.to(device)
            features, reconstructions, pred = classifier(b)
            result_features.append(features.cpu().detach())
            result_pred.append(pred.cpu().detach())
            print("cc: ",cc, "ds features:", features)

    result_pred = np.concatenate(result_pred)
    result_features = np.concatenate(result_features)
    labels = ds.targets
    img_files = names

    #  --------------------- storage embedings -----------------------------------
    print("labels:",labels)
    print('Score for REAL face: ', result_pred)

    list_for_storage = [p for p in zip(labels, result_pred, result_features, img_files)]
    import pickle

    #! mnist_feature_train_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_train_easy2easy_features.pkl'  # strange results - recomend to doublecheck experiments
    #! mnist_feature_val_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_val_easy2easy_features.pkl'
    #! mnist_feature_test_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_test_easy2easy_features.pkl'

    e2e_feature_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_train_easy2easy_features.pkl'
    e2e_feature_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_val_easy2easy_features.pkl'
    e2e_feature_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_test_easy2easy_features.pkl'

    h2h_600_feature_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_train_all2all_features.pkl'
    h2h_600_feature_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_val_all2all_features.pkl'
    h2h_600_feature_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_test_all2all_features.pkl'

    # h2h_600_4classes_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_train_all2all_features.pkl'
    # h2h_600_4classes_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_val_all2all_features.pkl'
    # h2h_600_4classes_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_test_all2all_features.pkl'

    e2h_feature_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_train_easy2all_features.pkl'
    e2h_feature_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_val_easy2all_features.pkl'
    e2h_feature_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_test_easy2all_features.pkl'

    with open(h2h_600_4classes_train_path, 'wb') as f:
        pickle.dump(list_for_storage, f, pickle.HIGHEST_PROTOCOL)

def flatten(xss):
    return [x for xs in xss for x in xs]

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import os
def ML_capsul_antispoof():
    """
    Run ML classifier based on embedings from Efficientnet
    """
    import pickle
    # ---------------MNIST based Embedings for easy antispoof dataset  ----------------------------------
    # mnist_feature_train_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_train_easy2easy_features.pkl'
    # mnist_feature_val_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_val_easy2easy_features.pkl'
    # mnist_feature_test_path10='/home/evgeniy/models/efficientnet/capsul_emb/mnist_test_easy2easy_features.pkl'

    # ---------------easy based capsule embedings  for easy dataset ---------embeding length: 32-------------------------
    e2e_feature_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_train_easy2easy_features.pkl'
    e2e_feature_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_val_easy2easy_features.pkl'
    e2e_feature_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_test_easy2easy_features.pkl'

    # ---------------ALL dataset based capsule embedings  for ALL dataset ----------------------------------
    h2h_600_feature_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_train_all2all_features.pkl'
    h2h_600_feature_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_val_all2all_features.pkl'
    h2h_600_feature_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_2_600_test_all2all_features.pkl'

    # ---------------4 classes capsule Embedings for ALL easy/mid/hard/real dataset with balance on train part ----------------------------------
    h2h_600_4classes_train_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_train_all2all_features.pkl'
    h2h_600_4classes_val_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_val_all2all_features.pkl'
    h2h_600_4classes_test_path='/home/evgeniy/models/efficientnet/capsul_emb/capsul_4classes_600_test_all2all_features.pkl'


    with open(h2h_600_4classes_train_path, 'rb') as f:
        train_features = pickle.load(f)
    with open(h2h_600_4classes_val_path, 'rb') as f:
        val_features = pickle.load(f)
    with open(h2h_600_4classes_test_path, 'rb') as f:
        test_features = pickle.load(f)

    # ---------------CNN Embedings only for easy part of antispoof dataset  ----------------------------------
    # feature_train_path='/home/evgeniy/models/efficientnet/train_easy_features.pkl'
    # feature_val_path='/home/evgeniy/models/efficientnet/val_easy_features.pkl'
    # feature_test_path='/home/evgeniy/models/efficientnet/test_easy_features.pkl'
    # ---------------CNN Embedings for ALL antispoof dataset  ----------------------------------
    feature_train_path='/home/evgeniy/models/efficientnet/train_features.pkl'
    feature_val_path='/home/evgeniy/models/efficientnet/val_features.pkl'
    feature_test_path_hard='/home/evgeniy/models/efficientnet/test_features.pkl'

    with open(feature_train_path, 'rb') as f:
        CNN_train_features = pickle.load(f)
    with open(feature_val_path, 'rb') as f:
        CNN_val_features = pickle.load(f)
    with open(feature_test_path_hard, 'rb') as f:
        CNN_test_features = pickle.load(f)


    X = []
    Y = []
    f_train = []
    Y_Capsule_predict = []
    #  --- use (train and val) for svm + PCA training ----
    for i in range(0,len(val_features)):
        X.append(flatten(val_features[i][2].tolist()) + CNN_val_features[i][1].tolist())    # Capsule + CNN
        # X.append(flatten(val_features[i][2].tolist()))                                    # Capsule only
        # X.append(CNN_val_features[i][1].tolist())                                         # CNN only
        # Y.append(val_features[i][0])                                                      # swicth for detection and classification
        if val_features[i][0] == 3:
            Y.append(1)
        else:
            Y.append(0)

        f_train.append(CNN_val_features[i][2])
        if val_features[i][1] == 3:
            Y_Capsule_predict.append(1)
        else:
            Y_Capsule_predict.append(0)

    print("val Capsule acc:", accuracy_score(Y, Y_Capsule_predict))

    for i in range(0, len(CNN_train_features)):
        fname = os.path.basename(CNN_train_features[i][2])[:-4]
        for j in range(0,len(train_features)):
            if fname in train_features[j][3]:
                X.append(flatten(train_features[j][2].tolist()) + CNN_train_features[i][1].tolist())  # Capsule + CNN
                # X.append(flatten(train_features[i][2].tolist()))                                    # Capsule only
                # X.append(CNN_train_features[i][1].tolist())                                         # CNN only
                # Y.append(train_features[j][0])                                                      # swicth for detection and classification
                if train_features[j][0] == 3:
                    Y.append(1)
                else:
                    Y.append(0)
                f_train.append(CNN_train_features[i][2])
                if train_features[j][1] == 3:
                    Y_Capsule_predict.append(1)
                else:
                    Y_Capsule_predict.append(0)

    print("train+val Capsule acc:", accuracy_score(Y, Y_Capsule_predict))

    X_test = []
    Y_test = []
    f_test = []
    Y_test_Capsule_predict = []

    for i in range(0,len(test_features)):
        X_test.append(flatten(test_features[i][2].tolist())+ CNN_test_features[i][1].tolist())  # Capsule + CNN
        # X_test.append(flatten(test_features[i][2].tolist()))                                  # Capsule only
        # X_test.append(CNN_test_features[i][1].tolist())                                       # CNN only
        # Y_test.append(test_features[i][0])                                                    # swicth for detection and classification
        if test_features[i][0] == 3:
            Y_test.append(1)
        else:
            Y_test.append(0)

        f_test.append(CNN_test_features[i][2])
        if test_features[i][1] == 3:
            Y_test_Capsule_predict.append(1)
        else:
            Y_test_Capsule_predict.append(0)

    print("test Capsule acc:", accuracy_score(Y_test, Y_test_Capsule_predict))


    clf = svm.SVC()
    clf.fit(X, Y)
    y_pred = clf.predict(X_test)
    y_pred_tr = clf.predict(X)
    print("ALL : acc: ", accuracy_score(Y_test, y_pred), ": train  acc: ", accuracy_score(Y, y_pred_tr))


    for ii in range(10, 100):
        pca = PCA(n_components=ii)

        clf = svm.SVC()

        pca.fit(X)
        X_pca = pca.transform(X)
        clf.fit(X_pca, Y)

        X_test_pca = pca.transform(X_test)
        y_pred = clf.predict(X_test_pca)
        y_pred_tr = clf.predict(X_pca)

        print(ii, ": acc: ",accuracy_score(Y_test, y_pred), ": train  acc: ",accuracy_score(Y, y_pred_tr))



if __name__ == '__main__':
    # main()                                #    Run the experiment with simple CNN
    # cnntest()                             #    Run embeding extraction from Efficientnet
    # ML_antispoof()                        #    Run ML classifier based on embedings from Efficientnet

    # capsule_test()                        #    Run embeding extraction from Capsule
    ML_capsul_antispoof()                   #    Run fussion:  ML classifier based on embedings from Capsule and Efficientnet

