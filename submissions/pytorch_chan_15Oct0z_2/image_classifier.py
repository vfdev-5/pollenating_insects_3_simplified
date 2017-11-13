from __future__ import division, print_function
import os, sys
from datetime import datetime
from glob import glob

import numpy as np
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
from torch.autograd import Variable
from torch.optim import Adam

from torchvision.models.resnet import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor


HAS_GPU = torch.cuda.is_available()
SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SIZE = (224, 224)
SEED = 12345

print("HAS_GPU: {}".format(HAS_GPU))


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet50PollenatingInsects(Module):

    def __init__(self):
        super(ResNet50PollenatingInsects, self).__init__()
        resnet = resnet50(pretrained=True)

        features = list(resnet.children())
        features = features[:-1]  # remove the last fc
        features.append(Flatten())
        self.features = Sequential(*features)

        # Final convolution is initialized differently form the rest
        self.classifier = Linear(resnet.fc.in_features, 403)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageClassifier(object):

    def __init__(self):

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.net = ResNet50PollenatingInsects()
        print_trainable_parameters(self.net)

        # self.c_weights = None
        self.c_weights = c_wts
        if HAS_GPU:
            self.net = self.net.cuda()
            self.c_weights = self.c_weights.cuda()

        self.batch_size = 48
        self.n_epochs = 15
        self.n_workers = 6
        self.n_splits = 10
        self.n_tta = 3
        self.lr = 0.000123
        self.lr_gamma = 0.789
        self.lr_step = None
        self._write_conf_log("{}".format(self.__dict__))

        #     validation_split = 0.1
        #     batch_size = 32
        #     nb_epochs = 12
        #     lr = 1e-4
        #     w_decay = 5e-4
        #     gamma = 0.5
        #     patience = 1

    def _get_train_aug(self):
        # http://pytorch.org/docs/master/torchvision/models.html
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        train_transforms = Compose([
            RandomCrop(SIZE),
            # Geometry
            RandomChoice([
                RandomAffine(rotation=(-90, 90), scale=(0.95, 1.05), translate=(0.05, 0.05)),
                RandomFlip(proba=0.5, mode='h'),
                RandomFlip(proba=0.5, mode='v'),
            ]),
            # To Tensor (float, CxHxW, [0.0, 1.0]) + Normalize
            ToTensor(),
            Normalize(mean_val, std_val)
        ])
        return train_transforms

    # def _transform(self, x):
    #     if x.shape[2] == 4:
    #         x = x[:, :, 0:3]
    #     # square center crop
    #     h, w = x.shape[:2]
    #     min_shape = min(h, w)
    #     x = x[h // 2 - min_shape // 2:h // 2 + min_shape // 2,
    #           w // 2 - min_shape // 2:w // 2 + min_shape // 2]
    #     # large resize & random crop
    #     x = resize(x, (320, 320), preserve_range=True)
    #     h1 = np.random.randint(96+1)
    #     w1 = np.random.randint(96+1)
    #     x = x[h1:h1+224, w1:w1+224]
    #     # random horizontal reflection
    #     if np.random.rand() > 0.5:
    #         x = x[:, ::-1, :]
    #     # random rotation
    #     angle = np.random.randint(-90, 90 + 1)
    #     x = rotate(x, angle, preserve_range=True)
    #     # HWC -> CHW
    #     x = x.transpose((2, 0, 1))
    #     # preprocess
    #     x /= 255.
    #     x[0, :, :] -= 0.485
    #     x[0, :, :] /= 0.229
    #     x[1, :, :] -= 0.456
    #     x[1, :, :] /= 0.224
    #     x[2, :, :] -= 0.406
    #     x[2, :, :] /= 0.225
    #     return x

    def _get_test_aug(self):
        # http://pytorch.org/docs/master/torchvision/models.html
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        test_transforms = Compose([
            CenterCrop(SIZE),
            # Geometry
            RandomChoice([
                # RandomAffine(rotation=(-60, 60), scale=(0.95, 1.05), translate=(0.05, 0.05)),
                RandomFlip(proba=0.5, mode='h'),
                RandomFlip(proba=0.5, mode='v'),
            ]),
            # To Tensor (float, CxHxW, [0.0, 1.0])  + Normalize
            ToTensor(),
            Normalize(mean_val, std_val)
        ])
        return test_transforms
        
    # def _transform_test(self, x):
    #     if x.shape[2] == 4:
    #         x = x[:, :, 0:3]
    #     # square center crop
    #     h, w = x.shape[:2]
    #     min_shape = min(h, w)
    #     x = x[h // 2 - min_shape // 2:h // 2 + min_shape // 2,
    #           w // 2 - min_shape // 2:w // 2 + min_shape // 2]
    #     # large resize & center crop
    #     x = resize(x, (320, 320), preserve_range=True)
    #     x = x[48:-48, 48:-48]
    #     # no random reflection
    #     # no random rotation
    #     # HWC -> CHW
    #     x = x.transpose((2, 0, 1))
    #     # preprocess
    #     x /= 255.
    #     x[0, :, :] -= 0.485
    #     x[0, :, :] /= 0.229
    #     x[1, :, :] -= 0.456
    #     x[1, :, :] /= 0.224
    #     x[2, :, :] -= 0.406
    #     x[2, :, :] /= 0.225
    #     return x

    # def _load_minibatch(self, img_loader, indexes):
    #     transforms = []
    #     X, y = img_loader.parallel_load(indexes, transforms)
    #     X = np.array([self._transform(x) for x in X], dtype=np.float32)
    #     X = _make_variable(X)
    #     y = np.array(y)
    #     y = _make_variable(y)
    #     return X, y
    #
    # def _load_val_minibatch(self, img_loader, indexes):
    #     transforms = []
    #     X, y = img_loader.parallel_load(indexes, transforms)
    #     X = np.array([self._transform_test(x) for x in X], dtype=np.float32)
    #     X = _make_variable(X)
    #     y = np.array(y)
    #     y = _make_variable(y)
    #     return X, y
    #
    # def _load_test_minibatch(self, img_loader, indexes):
    #     X = img_loader.parallel_load(indexes)
    #     X = np.array([self._transform_test(x) for x in X], dtype=np.float32)
    #     X = _make_variable(X)
    #     return X

    # def fit(self, img_loader):
    #     validation_split = 0.1
    #     batch_size = 32
    #     nb_epochs = 12
    #     lr = 1e-4
    #     w_decay = 5e-4
    #     gamma = 0.5
    #     patience = 1
    #
    #     criterion = nn.CrossEntropyLoss(weight=self.c_weights)
    #     criterion = criterion.cuda()
    #
    #     optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=w_decay)
    #     scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #
    #     val_acc_b = 0
    #     earlystcr = 0
    #
    #     for epoch in range(nb_epochs):
    #         t0 = time.time()
    #         self.net.train()  # train mode
    #         nb_trained = 0
    #         nb_updates = 0
    #         train_loss = []
    #         train_acc = []
    #         n_images = len(img_loader) * (1 - validation_split)
    #         n_images = int(n_images)
    #         i = 0
    #         while i < n_images:
    #             indexes = range(i, min(i + batch_size, n_images))
    #             X, y = self._load_minibatch(img_loader, indexes)
    #             i += len(indexes)
    #             # zero grad
    #             optimizer.zero_grad()
    #             # forward
    #             y_pred = self.net(X)
    #             loss = criterion(y_pred, y)
    #             # backward
    #             loss.backward()
    #             # update
    #             optimizer.step()
    #             # loss and accuracy
    #             train_acc.extend(self._get_acc(y_pred, y))
    #             train_loss.append(loss.data[0])
    #             nb_trained += X.size(0)
    #             nb_updates += 1
    #             if nb_updates % 100 == 0:
    #                 print(
    #                     'Epoch [{}/{}], [trained {}/{}], avg_loss: {:.4f}'
    #                     ', avg_train_acc: {:.4f}'.format(
    #                         epoch + 1, nb_epochs, nb_trained, n_images,
    #                         np.mean(train_loss), np.mean(train_acc)))
    #
    #         self.net.eval()  # eval mode
    #         valid_acc = []
    #         n_images = len(img_loader)
    #         while i < n_images:
    #             indexes = range(i, min(i + batch_size, n_images))
    #             X, y = self._load_val_minibatch(img_loader, indexes)
    #             i += len(indexes)
    #             y_pred = self.net(X)
    #             valid_acc.extend(self._get_acc(y_pred, y))
    #
    #         delta_t = time.time() - t0
    #         valid_acc_mean = np.mean(valid_acc)
    #
    #         print('Finished epoch {}'.format(epoch + 1))
    #         print('Time spent : {:.4f}'.format(delta_t))
    #         print('Train acc : {:.4f}'.format(np.mean(train_acc)))
    #         print('Valid acc : {:.4f}'.format(valid_acc_mean))
    #
    #         # decay lr
    #         scheduler.step()
    #
    #         # early stop?
    #         if valid_acc_mean > val_acc_b:
    #             val_acc_b = valid_acc_mean
    #             earlystcr = 0
    #         else:
    #             earlystcr += 1
    #         if earlystcr > patience:
    #             break

    # def _get_acc(self, y_pred, y_true):
    #     y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
    #     y_true = y_true.cpu().data.numpy()
    #     return (y_pred == y_true)

    # def predict_proba(self, img_loader):
    #     batch_size = 32
    #     n_images = len(img_loader)
    #     i = 0
    #     y_proba = np.empty((n_images, 403))
    #     while i < n_images:
    #         indexes = range(i, min(i + batch_size, n_images))
    #         X = self._load_test_minibatch(img_loader, indexes)
    #         i += len(indexes)
    #         y_proba[indexes] = nn.Softmax()(self.net(X)).cpu().data.numpy()
    #     return y_proba

    def _get_trainval_datasets(self, img_loader, n_splits=5, seed=12345, batch_size=32, num_workers=4):

        train_ds = ImageLoaderProxyDataset(img_loader)
        # Resize to 320x320
        train_ds = ResizedDataset(train_ds, (320, 320))
        # Stratified splits:
        n_samples = len(img_loader)
        X = np.zeros(n_samples)
        Y = np.zeros(n_samples, dtype=np.int)
        for i, label in enumerate(img_loader.y_array):
            Y[i] = label
        kfolds_train_indices = []
        kfolds_val_indices = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_indices, val_indices in skf.split(X, Y):
            kfolds_train_indices.append(train_indices)
            kfolds_val_indices.append(val_indices)
        kfold_samplers = []
        for train_indices, val_indices in zip(kfolds_train_indices, kfolds_val_indices):
            kfold_samplers.append({"train": SubsetRandomSampler(train_indices),
                                   "val": SubsetRandomSampler(val_indices)})
        # Data augmentations:
        train_transforms = self._get_train_aug()
        test_transforms = self._get_test_aug()
        # Transformed dataset
        data_aug_train_ds = TransformedDataset(train_ds, x_transforms=train_transforms)
        data_aug_val_ds = TransformedDataset(train_ds, x_transforms=test_transforms)

        # Dataloader prefetch + batching
        split_index = 0
        if HAS_GPU:
            train_batches_ds = OnGPUDataLoader(data_aug_train_ds,
                                               batch_size=batch_size,
                                               sampler=kfold_samplers[split_index]["train"],
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)
            val_batches_ds = OnGPUDataLoader(data_aug_val_ds,
                                             batch_size=batch_size,
                                             sampler=kfold_samplers[split_index]["val"],
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)
            return train_batches_ds, val_batches_ds

        train_batches_ds = DataLoader(data_aug_train_ds,
                                      batch_size=batch_size,
                                      sampler=kfold_samplers[split_index]["train"],
                                      num_workers=num_workers,
                                      drop_last=True)
        val_batches_ds = DataLoader(data_aug_val_ds,
                                    batch_size=batch_size,
                                    sampler=kfold_samplers[split_index]["val"],
                                    num_workers=num_workers,
                                    drop_last=True)
        return train_batches_ds, val_batches_ds

    def _get_test_datasets(self, img_loader, batch_size=32, num_workers=4):
        test_ds = ImageLoaderProxyDataset(img_loader, with_y=False)
        # Resize to 320x320
        test_ds = ResizedDataset(test_ds, (320, 320))
        # Data augmentations:
        test_transforms = self._get_test_aug()
        # Transformed dataset
        data_aug_test_ds = TransformedDataset(test_ds, x_transforms=test_transforms)
        # Dataloader prefetch + batching
        if HAS_GPU:
            test_batches_ds = OnGPUDataLoader(data_aug_test_ds,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True)
            return test_batches_ds

        test_batches_ds = DataLoader(data_aug_test_ds,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_batches_ds

    def _train_one_epoch(self, model, train_batches, criterion, optimizer, epoch, n_epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        model.train()
        try:

            with get_tqdm(total=len(train_batches)) as pbar:
                for i, (batch_x, batch_y) in enumerate(train_batches):

                    batch_x = Variable(batch_x)
                    batch_y = Variable(batch_y)
                    # compute output
                    batch_y_pred = model(batch_x)
                    loss = criterion(batch_y_pred, batch_y)
                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(batch_y_pred.data, batch_y.data, topk=(1, 5))
                    losses.update(loss.data[0], batch_x.size(0))
                    top1.update(prec1[0], batch_x.size(0))
                    top5.update(prec5[0], batch_x.size(0))
                    # compute gradient and do optimizer step
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    prefix_str = "Epoch: {}/{}".format(epoch + 1, n_epochs)
                    post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                                   'Prec@1 {top1.avg:.3f} | ' + \
                                   'Prec@5 {top5.avg:.3f}'
                    post_fix_str = post_fix_str.format(loss=losses, top1=top1, top5=top5)

                    if HAS_TQDM:
                        pbar.set_description_str(prefix_str, refresh=False)
                        pbar.set_postfix_str(post_fix_str, refresh=False)
                        pbar.update(1)
                    elif i % 100 == 0:
                        print(prefix_str, post_fix_str)

            return losses.avg, top1.avg, top5.avg
        except Exception as e:
            print("Exception caught: {}".format(e))
            return None, None, None

    def _validate(self, model, val_batches, criterion):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()
        try:
            with get_tqdm(total=len(val_batches)) as pbar:
                for i, (batch_x, batch_y) in enumerate(val_batches):
                    batch_x = Variable(batch_x, volatile=True)
                    batch_y = Variable(batch_y, volatile=True)
                    # compute output
                    batch_y_pred = model(batch_x)
                    loss = criterion(batch_y_pred, batch_y)
                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(batch_y_pred.data, batch_y.data, topk=(1, 5))
                    losses.update(loss.data[0], batch_x.size(0))
                    top1.update(prec1[0], batch_x.size(0))
                    top5.update(prec5[0], batch_x.size(0))

                    post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                                   'Prec@1 {top1.avg:.3f} | ' + \
                                   'Prec@5 {top5.avg:.3f}'
                    post_fix_str = post_fix_str.format(loss=losses, top1=top1, top5=top5)
                    if HAS_TQDM:
                        pbar.set_description_str("Validation", refresh=False)
                        pbar.set_postfix_str(post_fix_str, refresh=False)
                        pbar.update(1)
                    elif i % 100 == 0:
                        print("Validation: ", post_fix_str)

                return losses.avg, top1.avg, top5.avg
        except Exception as e:
            print("Exception caught: {}".format(e))
            return None, None, None

    def _save_checkpoint(self, state):
        best_model_filenames = glob(os.path.join(self.logs_path, 'model_val_prec1*'))
        for fn in best_model_filenames:
            os.remove(fn)
        best_model_filename='model_val_prec1={val_prec1:.4f}.pth.tar'.format(
            val_prec1=state['val_prec1']
        )
        torch.save(state, os.path.join(self.logs_path, best_model_filename))

    def _load_checkpoint(self, filename, model, optimizer=None):
        print("Load checkpoint: %s" % filename)
        state = torch.load(filename)
        model.load_state_dict(state['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
        return state

    def _write_csv_log(self, line):
        csv_file = os.path.join(self.logs_path, 'log.csv')
        self._write_log(csv_file, line)

    def _write_conf_log(self, line):
        conf_file = os.path.join(self.logs_path, 'conf.log')
        self._write_log(conf_file, line)

    def _write_log(self, filename, line):
        d = 'w' if not os.path.exists(filename) else 'a'
        with open(filename, d) as w:
            w.write(line + '\n')

    def _verbose_optimizer(self, optimizer):
        msg = "\nOptimizer parameters: \n"
        for pg in optimizer.param_groups:
            msg += "- Param group: \n"
            for k in pg:
                if k == 'params':
                    continue
                msg += "\t{}: {}\n".format(k, pg[k])
        return msg

    def fit(self, img_loader):

        best_model_filenames = glob(os.path.join(self.logs_path, 'model_val_prec1*'))
        if len(best_model_filenames) > 0 and os.path.exists(best_model_filenames[0]):
            print("Found weights : %s" % best_model_filenames[0])
            return

        # self.c_weights = torch.Tensor(compute_class_weight('balanced', np.arange(403), img_loader.y_array))
        # if HAS_GPU:
        #     self.c_weights = self.c_weights.cuda()

        batch_size = self.batch_size
        n_epochs = self.n_epochs
        num_workers = self.n_workers
        n_splits = self.n_splits

        lr = self.lr
        optimizer = Adam([{
            'params': self.net.features.parameters(),
            'lr': lr
        }, {
            'params': self.net.classifier.parameters(),
            'lr': 3.0*lr
        }])

        self._write_conf_log(self._verbose_optimizer(optimizer))

        criterion = nn.CrossEntropyLoss(weight=self.c_weights)
        if HAS_GPU:
            criterion = criterion.cuda()

        if self.lr_step is not None:
            # lr <- lr_init * gamma ** (epoch // step)
            scheduler = StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        else:
            # lr <- lr_init * gamma ** epoch
            scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)

        onplateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_batches_ds, val_batches_ds = self._get_trainval_datasets(img_loader,
                                                                       seed=SEED,
                                                                       n_splits=n_splits,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)
        self._write_csv_log("epoch,train_loss,train_prec1,val_loss,val_prec1")
        best_prec1 = 0.0
        for epoch in range(n_epochs):
            scheduler.step()
            # Verbose learning rates:
            print(self._verbose_optimizer(optimizer))

            # train for one epoch
            train_loss, train_prec1, _ = \
                self._train_one_epoch(self.net, train_batches_ds, criterion, optimizer, epoch, n_epochs)
            assert train_loss, train_prec1

            # evaluate on validation set
            val_loss, val_prec1, _ = self._validate(self.net, val_batches_ds, criterion)
            assert val_loss, val_prec1
            onplateau_scheduler.step(val_loss)

            # Write a csv log file
            self._write_csv_log("%i,%f,%f,%f,%f" % (epoch, train_loss, train_prec1, val_loss, val_prec1))

            # remember best prec@1 and save checkpoint
            if val_prec1 > best_prec1:
                best_prec1 = max(val_prec1, best_prec1)
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.net.state_dict(),
                    'val_prec1': val_prec1,
                    'optimizer': optimizer.state_dict()})

    def predict_proba(self, img_loader):

        # Load pretrained model
        best_model_filenames = glob(os.path.join(self.logs_path, 'model_val_prec1*'))
        assert len(best_model_filenames) > 0, "No pretrained models found"
        self._load_checkpoint(best_model_filenames[0], model=self.net)
        # switch to evaluate mode
        self.net.eval()

        batch_size = 32
        n_images = len(img_loader)
        y_probas = np.empty((self.n_tta, n_images, 403))

        for r in range(self.n_tta):
            print("- TTA round : %i / %i" % (r + 1, self.n_tta))
            test_batches_ds = self._get_test_datasets(img_loader, batch_size=batch_size)

            with get_tqdm(total=len(test_batches_ds)) as pbar:
                for i, (batch_x, batch_indices) in enumerate(test_batches_ds):
                    indexes = batch_indices.cpu().numpy()
                    batch_x = Variable(batch_x, volatile=True)
                    # compute output
                    batch_y_pred = self.net(batch_x)
                    y_probas[r, indexes, :] = nn.Softmax()(batch_y_pred).cpu().data.numpy()
                    if HAS_TQDM:
                        pbar.update(1)

        y_probas = np.mean(y_probas, axis=0)
        return y_probas


# *********************************************************************************
# PyTorch utils
# *********************************************************************************

class ProxyDataset(Dataset):

    def __init__(self, ds):
        assert isinstance(ds, Dataset)
        self.ds = ds

    def __len__(self):
        return len(self.ds)


class ResizedDataset(ProxyDataset):

    def __init__(self, ds, output_size, interpolation=cv2.INTER_CUBIC):
        super(ResizedDataset, self).__init__(ds)
        self.output_size = output_size
        self.interpolation = interpolation

    def __getitem__(self, index):
        x, y = self.ds[index]
        # RGBA -> RGB
        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        x = cv2.resize(x, dsize=self.output_size, interpolation=self.interpolation)
        return x, y


class CachedDataset(ProxyDataset):

    def __init__(self, ds, n_cached_images=10000):
        super(CachedDataset, self).__init__(ds)
        self.n_cached_images = n_cached_images
        self.cache = {}
        self.cache_hist = []

    def reset(self):
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            x, y = self.ds[index]
            if len(self.cache) > self.n_cached_images:
                first_index = self.cache_hist.pop(0)
                del self.cache[first_index]

            self.cache[index] = (x, y)
            self.cache_hist.append(index)
            return x, y


class TransformedDataset(ProxyDataset):

    def __init__(self, ds, x_transforms, y_transforms=None):
        super(TransformedDataset, self).__init__(ds)
        assert callable(x_transforms)
        if y_transforms is not None:
            assert callable(y_transforms)
        self.ds = ds
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

    def __getitem__(self, index):
        x, y = self.ds[index]
        x = self.x_transforms(x)
        if self.y_transforms is not None:
            y = self.y_transforms(y)

        return x, y


class ImageLoaderProxyDataset(Dataset):

    def __init__(self, img_loader, with_y=True):
        self.img_loader = img_loader
        self.with_y = with_y

        # test nb of channels:
        dp = img_loader.load(0)
        if isinstance(dp, tuple) and len(dp) == 2:
            x, _ = dp
            if with_y:
                self._to_dp = ImageLoaderProxyDataset._to_dp2
            else:
                self._to_dp = ImageLoaderProxyDataset._to_dp2_without_y
        else:
            x = dp
            self.with_y = False
            self._to_dp = ImageLoaderProxyDataset._to_dp1
        assert len(x.shape) == 3 and x.shape[-1] == 3, "x.shape={}".format(x.shape)

    @staticmethod
    def _to_dp2(load_fn, index):
        return load_fn(index)

    @staticmethod
    def _to_dp1(load_fn, index):
        return load_fn(index), index

    @staticmethod
    def _to_dp2_without_y(load_fn, index):
        return load_fn(index)[0], index

    def __len__(self):
        return len(self.img_loader)

    def __getitem__(self, index):
        return self._to_dp(self.img_loader.load, index)


class OnGPUDataLoaderIter(DataLoaderIter):

    def __next__(self):
        batch = super(OnGPUDataLoaderIter, self).__next__()
        cuda_batch = []
        for b in batch:
            if not b.is_pinned():
                b = b.pin_memory()
            cuda_batch.append(b.cuda(async=True))
        return cuda_batch

    next = __next__  # Python 2 compatibility


class OnGPUDataLoader(DataLoader):

    def __iter__(self):
        return OnGPUDataLoaderIter(self)


class RandomOrder(object):

    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        order = np.random.permutation(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(object):

    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        c = np.random.choice(len(self.transforms), 1)[0]
        return self.transforms[c](img)


class RandomFlip(object):

    def __init__(self, proba=0.5, mode='h'):
        assert mode in ['h', 'v']
        self.mode = mode
        self.proba = proba

    def __call__(self, img):
        if self.proba > np.random.rand():
            return img
        flipCode = 1 if self.mode == 'h' else 0
        return cv2.flip(img, flipCode)


class Crop(object):

    def __init__(self, size, padding=0):
        assert len(size) == 2
        self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        raise NotImplementedError()

    def __call__(self, img):
        if self.padding > 0:
            img = np.pad(img, self.padding, mode='edge')
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i + h, j:j + w, :]


class RandomCrop(Crop):

    @staticmethod
    def get_params(img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw


class CenterCrop(Crop):

    @staticmethod
    def get_params(img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw


class RandomAffine(object):

    def __init__(self, rotation=(-90, 90), scale=(0.85, 1.15), translate=(0.2, 0.2)):
        self.rotation = rotation
        self.scale = scale
        self.translate = translate

    def __call__(self, img):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        deg = np.random.uniform(self.rotation[0], self.rotation[1])
        max_dx = self.translate[0] * img.shape[1]
        max_dy = self.translate[1] * img.shape[0]
        dx = np.round(np.random.uniform(-max_dx, max_dx))
        dy = np.round(np.random.uniform(-max_dy, max_dy))
        center = (img.shape[1::-1] * np.array((0.5, 0.5))) - 0.5
        transform_matrix = cv2.getRotationMatrix2D(tuple(center), deg, scale)
        # Apply shift :
        transform_matrix[0, 2] += dx
        transform_matrix[1, 2] += dy
        ret = cv2.warpAffine(img, transform_matrix, img.shape[1::-1],
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


class RandomAdd(object):

    def __init__(self, proba=0.5, value=(-0, 0), per_channel=None):
        self.proba = proba
        self.per_channel = per_channel
        self.value = value

    def __call__(self, img):
        if self.proba > np.random.rand():
            return img
        out = img.copy()
        value = np.random.randint(self.value[0], self.value[1])
        if self.per_channel is not None and \
                        self.per_channel in list(range(img.shape[-1])):
            out[:, :, self.per_channel] = np.clip(out[:, :, self.per_channel] + value, 0, 255)
        else:
            out[:, :, :] = np.clip(out[:, :, :] + value, 0, 255)
        return out


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class AlphaLerp(object):

    def __init__(self, var):
        if isinstance(var, (tuple, list)):
            assert len(var) == 2
            self.min_val = var[0]
            self.max_val = var[1]
        else:
            self.min_val = 0
            self.max_val = var

    def get_alpha(self):
        return np.random.uniform(self.min_val, self.max_val)

    def get_end_image(self, img):
        raise NotImplementedError

    def __call__(self, img):
        return img.lerp(self.get_end_image(img), self.get_alpha())


class Saturation(AlphaLerp):

    def __init__(self, var):
        super(Saturation, self).__init__(var)

    def get_end_image(self, img):
        return Grayscale()(img)


class Brightness(AlphaLerp):

    def __init__(self, var):
        super(Brightness, self).__init__(var)

    def get_end_image(self, img):
        return img.new().resize_as_(img).zero_()


class Contrast(AlphaLerp):

    def __init__(self, var):
        super(Contrast, self).__init__(var)

    def get_end_image(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        return gs


class ColorJitter(RandomOrder):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        """
        :param brightness: int or tuple: (min, max) with min < max in [0.0, 1.0]
        :param contrast: int or tuple: (min, max) with min < max in [0.0, 1.0]
        :param saturation: int or tuple: (min, max) with min < max in [0.0, 1.0]
        """
        assert brightness or contrast or saturation
        transforms = []
        if brightness is not None:
            transforms.append(Brightness(brightness))
        if contrast is not None:
            transforms.append(Contrast(contrast))
        if saturation is not None:
            transforms.append(Saturation(saturation))
        super(ColorJitter, self).__init__(transforms)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.
    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
    )
    f = kwargs.get('file', sys.stderr)
    isatty = f.isatty()
    # Jupyter notebook should be recognized as tty. Wait for
    # https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(f, iostream.OutStream):
            isatty = True
    except ImportError:
        pass
    if isatty:
        default['mininterval'] = 0.25
    else:
        # If not a tty, don't refresh progress bar that often
        default['mininterval'] = 300
    default.update(kwargs)
    return default


def get_tqdm(**kwargs):
    """ Similar to :func:`get_tqdm_kwargs`,
    but returns the tqdm object directly. """
    if HAS_TQDM:
        return tqdm(**get_tqdm_kwargs(**kwargs))
    return FakeTqdm()


def print_trainable_parameters(m):
    total_number = 0
    for name, p in m.named_parameters():
        print(name, p.size())
        total_number += np.prod(p.size())
    print("\nTotal number of trainable parameters: ", total_number)


class FakeTqdm(object):
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


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
