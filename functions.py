import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# find the index of the classes to classify
def find_classes(path):
    classes = os.listdir(path)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


# get the indexes of the classes
def get_same_index(data, labels):
    idx_lab = []
    idx_my_data = []
    for i in labels:
        idx_lab.append(data.class_to_idx[i])
    for i in range(len(data)):
        if data[i][1] in idx_lab:
            idx_my_data.append(i)
    return np.array(idx_my_data)


# un-normalize image
def un_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor


# show batch of images
def imshow(img, mean, std, labels):
    print('Image labels : ', labels)
    img_ = img
    npimg = img_.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=False)
    img_ = un_normalize(img, mean, std)
    npimg = img_.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=False)


# parameter to set for requires_grad
def to_grad_param(model, feature_extract):
    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print('\t', name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t', name)
    return params_to_update


# set the parameters for the fine tuning to no grad require to keep learned features from pre-trained model
def set_parameter_requires_grad(model, feature_extracting):
    # if to keep the extracted features from the pre-trained model
    if feature_extracting:
        # for every parameter set to false
        for param in model.parameters():
            param.requires_grad = False
    return model


# test the accuracy of the model with the test set
def accuracy_test(model, test_loader):
    # model evaluation mode (requires_grad == False)
    model.eval()
    # initialization of correct labels
    correct = 0
    # initialization total
    total = 0
    # with no grad
    with torch.no_grad():
        # for every batch
        for data in test_loader:
            # take the data and the labels
            images, lab = data
            # model result
            outputs = model(images)
            # take the maximal probability
            _, predicted = torch.max(outputs.data, 1)
            # total number of samples
            total += lab.size(0)
            # add the number of correct labels and predictions
            correct += (predicted == lab).sum().item()
    print('Accuracy of the network : %d %%' % (100 * correct / total))
