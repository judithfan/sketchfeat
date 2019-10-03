import copy
import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
    
## feature dimensions by layer_ind
## 0: [64, 112, 112] = 802,816
## 1: [128, 56, 56] = 401,408
## 2: [256, 28, 28] = 200,704
## 3: [512, 14, 14] = 100,352
## 4: [512, 7, 7] = 50,176
## 5: [1, 4096]
## 6: [1, 4096]

use_cuda = torch.cuda.is_available()

class VGG19Embeddings(nn.Module):
    """Splits vgg19 into separate sections so that we can get
    feature embeddings from each section.
    :param vgg19: traditional vgg19 model
    """
    def __init__(self, vgg19, layer_index=-1, spatial_avg=True):
        super(VGG19Embeddings, self).__init__()
        self.conv1 = nn.Sequential(*(list(vgg19.features.children())[slice(0, 5)]))
        self.conv2 = nn.Sequential(*(list(vgg19.features.children())[slice(5, 10)]))
        self.conv3 = nn.Sequential(*(list(vgg19.features.children())[slice(10, 19)]))
        self.conv4 = nn.Sequential(*(list(vgg19.features.children())[slice(19, 28)]))
        self.conv5 = nn.Sequential(*(list(vgg19.features.children())[slice(28, 37)]))
        self.linear1 = nn.Sequential(*(list(vgg19.classifier.children())[slice(0, 2)]))
        self.linear2 = nn.Sequential(*(list(vgg19.classifier.children())[slice(3, 5)]))
        self.linear3 = nn.Sequential(list(vgg19.classifier.children())[-1])
        layer_index = int(float(layer_index)) # bll
        assert layer_index >= -1 and layer_index < 8
        self.layer_index = layer_index
        self.spatial_avg = spatial_avg

    def _flatten(self, x):
        if (self.spatial_avg==True) & (self.layer_index<5):
            x = x.mean(3).mean(2)
        return x.view(x.size(0), -1)

    def forward(self, x):
        # build in this ugly way so we don't have to evaluate things we don't need to.
        x_conv1 = self.conv1(x)
        if self.layer_index == 0:
            return self._flatten(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        if self.layer_index == 1:
            return self._flatten(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        if self.layer_index == 2:
            return self._flatten(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        if self.layer_index == 3:
            return self._flatten(x_9Bconv4)
        x_conv5 = self.conv5(x_conv4)
        x_conv5_flat = self._flatten(x_conv5)
        if self.layer_index == 4:
            return x_conv5_flat
        x_linear1 = self.linear1(x_conv5_flat)

        ## THIS IS FC6
        if self.layer_index == 5:
            return x_linear1
        x_linear2 = self.linear2(x_linear1)
        if self.layer_index == 6:
            return x_linear2
        # x_linear3 = self.linear3(x_linear2)
        # if self.layer_index == 7:
        #     return [x_linear3]
        # return [self._flatten(x_conv1), self._flatten(x_conv2),
        #         self._flatten(x_conv3), self._flatten(x_conv4),
        #         self._flatten(x_conv5), x_linear1, x_linear2, x_linear3]

class FeatureExtractor():
    def __init__(self, paths, layer=6, use_cuda=True, imsize=224, 
                 batch_size=64, cuda_device=2, data_type='images',
                 spatial_avg=True):
        self.layer = layer
        self.paths = paths
        self.num_images = len(self.paths)
        self.use_cuda = use_cuda
        self.imsize = imsize
        self.padding = 10
        self.batch_size = batch_size
        self.cuda_device = torch.device('cuda:{}'.format(cuda_device))
        self.data_type = data_type ## either 'images' or 'sketches'
        self.spatial_avg = spatial_avg ## if true, collapse across spatial dimensions to just preserve channel activation

    def flatten_list(self, x):
        return np.array([item for sublist in x for item in sublist])

    def load_image(self, image_path):
        # Image preprocessing, normalization for the pretrained resnet
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image
    
    def load_vgg19(self, layer_index):
        vgg19 = models.vgg19(pretrained = True).to(self.cuda_device)
        vgg19 = VGG19Embeddings(vgg19, layer_index, spatial_avg = self.spatial_avg)
        vgg19.eval()  # freeze dropout
        
        self.transform = transforms.Compose([
            transforms.Pad(self.padding),
            transforms.CenterCrop(self.imsize),
            transforms.Resize(self.imsize),
            transforms.ToTensor()])
        
        return vgg19
        
    def generator(self, paths):
        for path in paths:
            image = self.load_image(path)
            yield image, path

    def extract_feature_matrix(self):

        # load appropriate extractor
        extractor = self.load_vgg19(self.layer)

        # define generator
        generator = self.generator(self.paths)

        # initialize sketch and label matrices
        features = []
        paths = []
        n = 0
        quit = False

        # generate batches of sketches and labels
        if generator:
            while True:
                batch_size = self.batch_size
                img_batch = torch.zeros(batch_size, 3, self.imsize, self.imsize)
                paths_batch = []
                if use_cuda:
                    img_batch = img_batch.to(self.cuda_device)

                if (n+1)%20==0:
                    print('Batch {}'.format(n + 1))

                for b in range(batch_size):
                    try:
                        img, path = next(generator)
                        img_batch[b] = img
                        paths_batch.append(path)

                    except StopIteration:
                        quit = True
                        print('stopped!')
                        break

                n = n + 1
                if n == self.num_images//self.batch_size:
                    img_batch = img_batch.narrow(0,0,b)
                    paths_batch = paths_batch[:b + 1]

                # extract features from batch
                feats_batch = extractor(img_batch)
                feats_batch = feats_batch.cpu().data.numpy()

                if len(features)==0:
                    features = feats_batch
                else:
                    features = np.vstack((features,feats_batch))
                
                paths.append(paths_batch)
                if n == self.num_images//batch_size + 1:
                    break
#        paths = map(flatten_list, paths_batch)
        return features, paths
