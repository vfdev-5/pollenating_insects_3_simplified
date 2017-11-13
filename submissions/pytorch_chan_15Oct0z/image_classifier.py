from __future__ import division

import time
import math

import numpy as np

from skimage.transform import resize
from skimage.transform import rotate

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.optim import lr_scheduler

from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet152

from rampwf.workflows.image_classifier import get_nb_minibatches

is_cuda = torch.cuda.is_available()


def _make_variable(X):
    X = Variable(torch.from_numpy(X))
    if is_cuda:
        X = X.cuda()
    return X


def _flatten(x):
    return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, net='r18', pretrained=True):
        super(Net, self).__init__()
        
        if net == 'r18':
            resnet = resnet18(pretrained=True)
            expansion = 1
        elif net == 'r34':
            resnet = resnet34(pretrained=True)
            expansion = 1
        elif net == 'r50':
            resnet = resnet50(pretrained=True)
            expansion = 4
        elif net == 'r152':
            resnet = resnet152(pretrained=True)
            expansion = 4
        else:
            print('Unknown ResNet!')
            quit()
        
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        self.fc = nn.Linear(512 * expansion, 403)

    def forward(self, x):
        x = self.features(x)
        x = _flatten(x)
        x = self.fc(x)
        return x


class ImageClassifier(object):

    def __init__(self):
        self.net = Net(net='r50')
        self.c_weights = c_wts
        print ('CUDA: {}'.format(is_cuda))
        if is_cuda:
            self.net = self.net.cuda()
            self.c_weights = self.c_weights.cuda()
        print('Built net.')
        
    def _transform(self, x):
        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        # square center crop    
        h, w = x.shape[:2]
        min_shape = min(h, w)
        x = x[h // 2 - min_shape // 2:h // 2 + min_shape // 2,
              w // 2 - min_shape // 2:w // 2 + min_shape // 2]
        # large resize & random crop
        x = resize(x, (320, 320), preserve_range=True)
        h1 = np.random.randint(96+1)
        w1 = np.random.randint(96+1)
        x = x[h1:h1+224, w1:w1+224]
        # random horizontal reflection
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :]
        # random rotation
        angle = np.random.randint(-90, 90 + 1)
        x = rotate(x, angle, preserve_range=True)
        # HWC -> CHW
        x = x.transpose((2, 0, 1))
        # preprocess
        x /= 255.
        x[0, :, :] -= 0.485
        x[0, :, :] /= 0.229
        x[1, :, :] -= 0.456
        x[1, :, :] /= 0.224
        x[2, :, :] -= 0.406
        x[2, :, :] /= 0.225
        return x
        
    def _transform_test(self, x):
        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        # square center crop 
        h, w = x.shape[:2]
        min_shape = min(h, w)
        x = x[h // 2 - min_shape // 2:h // 2 + min_shape // 2,
              w // 2 - min_shape // 2:w // 2 + min_shape // 2]
        # large resize & center crop
        x = resize(x, (320, 320), preserve_range=True)
        x = x[48:-48, 48:-48]
        # no random reflection
        # no random rotation
        # HWC -> CHW
        x = x.transpose((2, 0, 1))
        # preprocess
        x /= 255.
        x[0, :, :] -= 0.485
        x[0, :, :] /= 0.229
        x[1, :, :] -= 0.456
        x[1, :, :] /= 0.224
        x[2, :, :] -= 0.406
        x[2, :, :] /= 0.225
        return x

    def _load_minibatch(self, img_loader, indexes):
        transforms = []
        X, y = img_loader.parallel_load(indexes, transforms)
        X = np.array([self._transform(x) for x in X], dtype=np.float32)
        X = _make_variable(X)
        y = np.array(y)
        y = _make_variable(y)
        return X, y

    def _load_val_minibatch(self, img_loader, indexes):
        transforms = []
        X, y = img_loader.parallel_load(indexes, transforms)
        X = np.array([self._transform_test(x) for x in X], dtype=np.float32)
        X = _make_variable(X)
        y = np.array(y)
        y = _make_variable(y)
        return X, y

    def _load_test_minibatch(self, img_loader, indexes):
        X = img_loader.parallel_load(indexes)
        X = np.array([self._transform_test(x) for x in X], dtype=np.float32)
        X = _make_variable(X)
        return X

    def fit(self, img_loader):
        validation_split = 0.1
        batch_size = 32
        nb_epochs = 12
        lr = 1e-4
        w_decay = 5e-4
        gamma = 0.5
        patience  = 1
        
        criterion = nn.CrossEntropyLoss(weight=self.c_weights)
        if is_cuda:
            criterion = criterion.cuda()

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=w_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        val_acc_b = 0        
        earlystcr = 0
        
        for epoch in range(nb_epochs):
            t0 = time.time()
            self.net.train()  # train mode
            nb_trained = 0
            nb_updates = 0
            train_loss = []
            train_acc = []
            n_images = len(img_loader) * (1 - validation_split)
            n_images = int(n_images)
            i = 0
            while i < n_images:
                indexes = range(i, min(i + batch_size, n_images))
                X, y = self._load_minibatch(img_loader, indexes)
                i += len(indexes)
                # zero grad
                optimizer.zero_grad()
                # forward
                y_pred = self.net(X)
                loss = criterion(y_pred, y)
                # backward
                loss.backward()
                # update
                optimizer.step()
                # loss and accuracy
                train_acc.extend(self._get_acc(y_pred, y))
                train_loss.append(loss.data[0])
                nb_trained += X.size(0)
                nb_updates += 1
                if nb_updates % 100 == 0:
                    print(
                        'Epoch [{}/{}], [trained {}/{}], avg_loss: {:.4f}'
                        ', avg_train_acc: {:.4f}'.format(
                            epoch + 1, nb_epochs, nb_trained, n_images,
                            np.mean(train_loss), np.mean(train_acc)))

            self.net.eval()  # eval mode
            valid_acc = []
            n_images = len(img_loader)
            while i < n_images:
                indexes = range(i, min(i + batch_size, n_images))
                X, y = self._load_val_minibatch(img_loader, indexes)
                i += len(indexes)
                y_pred = self.net(X)
                valid_acc.extend(self._get_acc(y_pred, y))

            delta_t = time.time() - t0
            valid_acc_mean = np.mean(valid_acc)
            
            print('Finished epoch {}'.format(epoch + 1))
            print('Time spent : {:.4f}'.format(delta_t))
            print('Train acc : {:.4f}'.format(np.mean(train_acc)))
            print('Valid acc : {:.4f}'.format(valid_acc_mean))
            
            # decay lr
            scheduler.step()
            
            # early stop?
            if valid_acc_mean > val_acc_b:
                val_acc_b = valid_acc_mean
                earlystcr = 0
            else:
                earlystcr += 1
            if earlystcr > patience:
                break

    def _get_acc(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
        y_true = y_true.cpu().data.numpy()
        return (y_pred == y_true)

    def predict_proba(self, img_loader):
        batch_size = 32
        n_images = len(img_loader)
        i = 0
        y_proba = np.empty((n_images, 403))
        while i < n_images:
            indexes = range(i, min(i + batch_size, n_images))
            X = self._load_test_minibatch(img_loader, indexes)
            i += len(indexes)
            y_proba[indexes] = nn.Softmax()(self.net(X)).cpu().data.numpy()
        return y_proba


c_wts = torch.Tensor([ 0.38884651,  0.38495664,  0.57723597,  0.34313049,  0.53840656,
        0.36387612,  0.58859191,  0.36206594,  0.40138692,  0.46595542,
        0.46266994,  0.4633142 ,  0.53331591,  0.57202085,  0.39582965,
        0.33416278,  0.50800542,  0.51967508,  0.40761119,  0.458328  ,
        0.35163532,  0.5799536 ,  0.43972321,  0.54381842,  0.45773055,
        0.38367371,  0.4443615 ,  0.31952047,  0.56469993,  0.56469993,
        0.51298098,  0.53667565,  0.56469993,  0.38745799,  0.45204123,
        0.34767881,  0.56469993,  0.48986166,  0.43796236,  0.35627025,
        0.46595542,  0.44062318,  0.40994818,  0.52108354,  0.44834024,
        0.51967508,  0.54017312,  0.5285148 ,  0.43883639,  0.432572  ,
        0.41702461,  0.4434054 ,  0.42308129,  0.48095855,  0.39045701,
        0.35670244,  0.50220315,  0.28629505,  0.43883639,  0.48095855,
        0.55791377,  0.4453326 ,  0.44782881,  0.49081525,  0.47610647,
        0.48351554,  0.37246093,  0.44937574,  0.50220315,  0.44681832,
        0.45258801,  0.48892034,  0.35541895,  0.4182464 ,  0.43296975,
        0.33794728,  0.40138692,  0.38273353,  0.44153662,  0.37385118,
        0.49785916,  0.50563183,  0.40660448,  0.54569994,  0.33703172,
        0.58274945,  0.55575916,  0.46396478,  0.45773055,  0.4443615 ,
        0.44293285,  0.54017312,  0.37829313,  0.57202085,  0.30712571,
        0.49276042,  0.56469993,  0.32696557,  0.37608621,  0.43710081,
        0.40322468,  0.59164733,  0.39542623,  0.42077325,  0.51045374,
        0.53667565,  0.49577763,  0.39939441,  0.35766914,  0.58859191,
        0.54017312,  0.54381842,  0.36286118,  0.55791377,  0.51045374,
        0.53840656,  0.54017312,  0.48707345,  0.54017312,  0.4514991 ,
        0.41465908,  0.34142421,  0.38973473,  0.51967508,  0.37886157,
        0.52546123,  0.46266994,  0.50332909,  0.44834024,  0.4091578 ,
        0.4135134 ,  0.31113406,  0.50109359,  0.42208058,  0.30147559,
        0.56951667,  0.38120662,  0.58859191,  0.54569994,  0.45713853,
        0.37436739,  0.53497915,  0.51170722,  0.55791377,  0.50220315,
        0.59805112,  0.50109359,  0.46203187,  0.34997677,  0.48265297,
        0.36945115,  0.37184275,  0.47610647,  0.51298098,  0.4514991 ,
        0.43458799,  0.55159834,  0.43417928,  0.48095855,  0.26972462,
        0.3713542 ,  0.49577763,  0.42583909,  0.34909971,  0.49276042,
        0.46595542,  0.48616736,  0.37436739,  0.38814732,  0.53008477,
        0.33041104,  0.47084268,  0.41793843,  0.44681832,  0.54958827,
        0.34497924,  0.55575916,  0.46663233,  0.51692985,  0.35272671,
        0.34852535,  0.46396478,  0.40275913,  0.45096155,  0.51045374,
        0.40138692,  0.4613999 ,  0.50332909,  0.35550331,  0.51045374,
        0.58859191,  0.41156459,  0.49178141,  0.42513708,  0.53497915,
        0.47012258,  0.46203187,  0.39212175,  0.4443615 ,  0.56707731,
        0.34997677,  0.46731612,  0.2642076 ,  0.51967508,  0.56469993,
        0.49375258,  0.4110204 ,  0.47012258,  0.49892199,  1.        ,
        0.83048202,  0.60141049,  0.89771172,  0.63299809,  0.66438562,
        0.67052815,  0.70672709,  0.69100954,  1.        ,  0.96025257,
        1.18329466,  1.04795164,  1.18329466,  1.28509721,  1.04795164,
        0.6425486 ,  0.67699249,  0.7563042 ,  0.67052815,  0.79663977,
        0.59164733,  0.61219425,  0.61219425,  0.74492186,  0.83048202,
        0.72452678,  0.64763985,  1.04795164,  0.68380838,  0.70672709,
        0.62850999,  1.18329466,  0.76862179,  0.61219425,  0.60141049,
        0.96025257,  1.18329466,  0.79663977,  1.18329466,  0.61604832,
        0.96025257,  1.28509721,  0.78201148,  1.28509721,  1.18329466,
        0.64763985,  0.70672709,  0.92662841,  0.85027415,  1.43067656,
        0.68380838,  0.63299809,  0.89771172,  1.10730936,  0.73436114,
        1.18329466,  0.78201148,  1.18329466,  0.70672709,  0.79663977,
        0.64763985,  1.        ,  0.73436114,  1.10730936,  0.87250287,
        0.74492186,  0.68380838,  0.72452678,  0.92662841,  0.79663977,
        0.69100954,  0.92662841,  0.89771172,  1.18329466,  1.18329466,
        0.60488291,  1.28509721,  0.79663977,  0.67052815,  0.69100954,
        0.78201148,  0.96025257,  0.62850999,  0.66438562,  0.72452678,
        1.28509721,  1.28509721,  0.65296361,  0.6425486 ,  1.28509721,
        0.68380838,  0.63299809,  0.59164733,  1.        ,  0.69863442,
        1.        ,  0.70672709,  1.04795164,  0.62850999,  0.65296361,
        0.63767307,  0.87250287,  0.76862179,  0.60488291,  0.66438562,
        0.69100954,  0.62850999,  1.18329466,  1.10730936,  1.18329466,
        0.89771172,  0.92662841,  0.89771172,  0.67699249,  0.60141049,
        0.87250287,  0.61219425,  1.10730936,  1.18329466,  0.78201148,
        0.62004589,  0.70672709,  1.        ,  0.87250287,  1.28509721,
        1.28509721,  0.68380838,  0.76862179,  1.18329466,  0.7563042 ,
        0.83048202,  1.10730936,  0.79663977,  0.83048202,  0.78201148,
        1.28509721,  1.        ,  0.65296361,  0.64763985,  1.18329466,
        0.62850999,  1.28509721,  0.76862179,  0.69863442,  0.7563042 ,
        0.96025257,  1.18329466,  0.61219425,  0.59164733,  0.7563042 ,
        0.92662841,  0.60488291,  0.72452678,  0.87250287,  0.81271151,
        0.61604832,  1.18329466,  0.96025257,  0.72452678,  1.10730936,
        0.64763985,  0.79663977,  0.83048202,  0.61219425,  1.18329466,
        1.10730936,  0.65853857,  1.04795164,  0.92662841,  0.69100954,
        0.67052815,  0.69863442,  0.78201148,  1.10730936,  0.83048202,
        1.10730936,  0.92662841,  1.28509721,  0.68380838,  0.65853857,
        0.73436114,  0.60141049,  0.65853857,  1.04795164,  0.76862179,
        1.18329466,  1.04795164,  0.65296361,  0.62004589,  0.74492186,
        0.62850999,  0.79663977,  0.92662841])
