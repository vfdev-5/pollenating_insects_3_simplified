from __future__ import division
import os, sys
from datetime import datetime
import shutil
from glob import glob

import numpy as np
import cv2

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn import Linear, Module, Sequential, AdaptiveAvgPool2d
from torch.autograd import Variable
from torch.optim import Adam

from torchvision.models import densenet161
from torchvision.transforms import Compose, Normalize, ToTensor


# HAS_GPU = torch.cuda.is_available()
HAS_GPU = False
SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SIZE = (452, 452)
SEED = 12345

print("HAS_GPU: {}".format(HAS_GPU))


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DenseNet161PollenatingInsects(Module):

    def __init__(self):
        super(DenseNet161PollenatingInsects, self).__init__()
        densenet = densenet161(pretrained=True)

        self.features = densenet.features

        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(densenet.classifier.in_features, 403)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageClassifier(object):

    def __init__(self):
        now = datetime.now()
        self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.net = DenseNet161PollenatingInsects()
        if HAS_GPU:
            self.net = self.net.cuda()

        self.batch_size = 12
        self.n_epochs = 1
        self.n_workers = 15
        self.n_splits = 7

    def _get_train_aug(self):
        # http://pytorch.org/docs/master/torchvision/models.html
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        train_transforms = Compose([
            RandomCrop(SIZE),
            # Geometry
            RandomChoice([
                RandomAffine(rotation=(-60, 60), scale=(0.95, 1.05), translate=(0.05, 0.05)),
                RandomFlip(proba=0.5, mode='h'),
                RandomFlip(proba=0.5, mode='v'),
            ]),
            # Color
            # RandomChoice([
            #    RandomAdd(value=(-10, 10), per_channel=0),
            #    RandomAdd(value=(-10, 10), per_channel=1),
            #    RandomAdd(value=(-10, 10), per_channel=2),
            #    RandomAdd(value=(-10, 10))
            # ]),
            # To Tensor (float, CxHxW, [0.0, 1.0]) + Normalize
            ToTensor(),
            Normalize(mean_val, std_val)
        ])
        return train_transforms

    def _get_test_aug(self):
        # http://pytorch.org/docs/master/torchvision/models.html
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        test_transforms = Compose([
            RandomCrop(SIZE),
            # Geometry
            RandomChoice([
                RandomAffine(rotation=(-60, 60), scale=(0.95, 1.05), translate=(0.05, 0.05)),
                RandomFlip(proba=0.5, mode='h'),
                RandomFlip(proba=0.5, mode='v'),
            ]),
            # Color
            # RandomAdd(value=(-10, 10)),
            # To Tensor (float, CxHxW, [0.0, 1.0])  + Normalize
            ToTensor(),
            Normalize(mean_val, std_val)
        ])
        return test_transforms

    def _get_trainval_datasets(self, img_loader, n_splits=5, seed=12345, batch_size=32, num_workers=4):
        train_ds = ImageLoaderProxyDataset(img_loader)
        # Resize to 512x512
        train_ds = ResizedDataset(train_ds, (512, 512))
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
        # Resize to 512 x 512
        test_ds = ResizedDataset(test_ds, (512, 512))
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
                    pbar.set_description_str(prefix_str, refresh=False)
                    post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                                   'Prec@1 {top1.avg:.3f} | ' + \
                                   'Prec@5 {top5.avg:.3f}'
                    post_fix_str = post_fix_str.format(loss=losses, top1=top1, top5=top5)
                    pbar.set_postfix_str(post_fix_str, refresh=False)
                    pbar.update(1)
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
                    batch_y = Variable(batch_y, volatile=True)   # see http://pytorch.org/docs/master/autograd.html#variable
                    # compute output
                    batch_y_pred = model(batch_x)
                    loss = criterion(batch_y_pred, batch_y)
                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(batch_y_pred.data, batch_y.data, topk=(1, 5))
                    losses.update(loss.data[0], batch_x.size(0))
                    top1.update(prec1[0], batch_x.size(0))
                    top5.update(prec5[0], batch_x.size(0))
                    pbar.set_description_str("Validation", refresh=False)
                    post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                                   'Prec@1 {top1.avg:.3f} | ' + \
                                   'Prec@5 {top5.avg:.3f}'
                    post_fix_str = post_fix_str.format(loss=losses, top1=top1, top5=top5)
                    pbar.set_postfix_str(post_fix_str, refresh=False)
                    pbar.update(1)
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
        state = torch.load(filename)
        model.load_state_dict(state['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
        return state

    def _write_csv_log(self, line):
        csv_file = os.path.join(self.logs_path, 'log.csv')
        d = 'x' if not os.path.exists(csv_file) else 'a'
        with open(csv_file, d) as w:
            w.write(line + '\n')

    def fit(self, img_loader):

        batch_size = self.batch_size
        n_epochs = self.n_epochs
        num_workers = self.n_workers
        n_splits = self.n_splits
        lr = 0.00005

        optimizer = Adam([{
                        'params': self.net.features.parameters(),
                        'lr': lr
                        }, {
                        'params': self.net.classifier.parameters(),
                        'lr': 10*lr
                        }])
        criterion = nn.CrossEntropyLoss()
        if HAS_GPU:
            criterion = criterion.cuda()
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        onplateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

        train_batches_ds, val_batches_ds = self._get_trainval_datasets(img_loader,
                                                                       seed=SEED,
                                                                       n_splits=n_splits,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)
        self._write_csv_log("Epoch;train_loss;train_prec1;val_loss;val_prec1")
        best_prec1 = 0.0
        for epoch in range(n_epochs):
            scheduler.step()
            # Verbose learning rates:

            # train for one epoch
            train_loss, train_prec1, _ = \
                self._train_one_epoch(self.net, train_batches_ds, criterion, optimizer, epoch, n_epochs)
            # evaluate on validation set
            val_loss, val_prec1, _ = self._validate(self.net, val_batches_ds, criterion)
            onplateau_scheduler.step(val_loss)
            # Write a csv log file
            self._write_csv_log("%i;%f;%f;%f;%f" % (epoch, train_loss, train_prec1, val_loss, val_prec1))

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

        n = 10
        y_probas = np.empty((n, n_images, 403))

        for r in range(n):
            print("- TTA round : %i" % r)
            test_batches_ds = self._get_test_datasets(img_loader, batch_size=batch_size)

            with get_tqdm(total=len(test_batches_ds)) as pbar:
                for i, (batch_x, batch_indices) in enumerate(test_batches_ds):
                    indexes = batch_indices.cpu().numpy()
                    batch_x = Variable(batch_x, volatile=True)
                    # compute output
                    batch_y_pred = self.net(batch_x)
                    y_probas[r, indexes, :] = nn.Softmax()(batch_y_pred).cpu().data.numpy()
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
        else:
            x = dp
            self.with_y = False
        assert len(x.shape) == 3 and x.shape[-1] == 3, "x.shape={}".format(x.shape)

    def __len__(self):
        return len(self.img_loader)

    def __getitem__(self, index):
        if self.with_y:
            return self.img_loader.load(index)
        return self.img_loader.load(index), index


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


class RandomOrder:

    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        order = np.random.permutation(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice:

    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        c = np.random.choice(len(self.transforms), 1)[0]
        return self.transforms[c](img)


class RandomFlip:

    def __init__(self, proba=0.5, mode='h'):
        assert mode in ['h', 'v']
        self.mode = mode
        self.proba = proba

    def __call__(self, img):
        if self.proba > np.random.rand():
            return img
        flipCode = 1 if self.mode == 'h' else 0
        return cv2.flip(img, flipCode)


class RandomCrop:

    def __init__(self, size, padding=0):
        assert len(size) == 2
        self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = np.pad(img, self.padding, mode='edge')
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i + h, j:j + w, :]


class RandomAffine:

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


class RandomAdd:

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


class AverageMeter:
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
    return tqdm(**get_tqdm_kwargs(**kwargs))
