import numpy as np
from copy import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def load_all_NN():

    import torchvision.models as models

    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)

    return resnet18,alexnet,squeezenet,vgg16,densenet,inception

def load_imagenet():
    import os
    import torchvision.datasets as datasets
    # Data loading code
    valdir = os.path.join("../","val/")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                              shuffle=True, num_workers=2)
    return valloader

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0., 0., 0.],
                              std=[1./255., 1./255., 1./255.])])


    trainset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    return testloader


def perturbator_3001(NeuralNet,sample_loader, pop_size=400,max_iterations=100, f_param=0.5,criterium="paper",pixel_number=1):

    list_pert_samples=[]
    list_iterations=[]

    soft=nn.Softmax()

    sample_count = 0

    for data,targets in sample_loader:

        #draw the coordinates and rgb-values of the agents from random
        coords_old=np.random.random_integers(0,data.size()[-1], (pop_size,pixel_number,2))
        rgb_old=np.random.normal(128,127,(pop_size,pixel_number, 3))
        #initialize agents of next iteration
        coords_new = np.zeros((pop_size,pixel_number,2))
        rgb_new = np.zeros((pop_size, pixel_number, 3))
        #set iterator to zero
        iteration = 0
        found_candidate = False
        data_purb = 0

        while iteration < 100:

            for i in range(pop_size):

                data_purb=data.clone()
                data_purb[0, :, coords_old[i, :, 0], coords_old[i, :, 1]] += torch.tensor(rgb_old[i].transpose(), dtype=torch.float)

                # softmax
                score = soft(NeuralNet(data_purb))

                true_score = score[0, targets]

                if true_score < 0.05:
                    found_candidate = True
                    list_iterations.append(iteration)
                    list_pert_samples.append(data_purb)
                    break

                #DE update agents
                r1, r2, r3 = np.random.choice(range(pop_size), 3, replace=False)
                coords_new[i] = coords_old[r1] + f_param*(coords_old[r2] + coords_old[r3])
                rgb_new[i] = rgb_old[r1] + f_param * (rgb_old[r2] + rgb_old[r3])

            if found_candidate:
                break

            coords_old = coords_new.copy()
            rgb_old = rgb_new.copy()

            iteration += 1

        if found_candidate == False:
            list_pert_samples.append(data_purb)
            list_iterations.append(iteration)

        sample_count += 1
        if sample_count>600:
            break

    return list_pert_samples, list_iterations


resnet18, alexnet, squeezenet, vgg16, densenet, inception = load_all_NN()
data=load_imagenet()
pert_samples, iterations = perturbator_3001(alexnet,data,5)
print("hi")