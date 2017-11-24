from __future__ import division, print_function
import os, sys
from datetime import datetime
from glob import glob

import numpy as np
import cv2

from sklearn.model_selection import StratifiedShuffleSplit

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn import Module
from torch.autograd import Variable
from torch.optim import Adam

from torchvision.models import squeezenet1_1
from torchvision.transforms import Compose, Normalize, ToTensor

from imblearn.over_sampling import RandomOverSampler


HAS_GPU = torch.cuda.is_available()
SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SIZE = (299, 299)
SEED = 12345

print("HAS_GPU: {}".format(HAS_GPU))


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SqueezeNetPollenatingInsects(Module):

    def __init__(self):
        super(SqueezeNetPollenatingInsects, self).__init__()
        squeezenet = squeezenet1_1(pretrained=True)

        self.features = squeezenet.features

        # Final convolution is initialized differently form the rest
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 403, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13),
            Flatten()
        )

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
        self.net = SqueezeNetPollenatingInsects()
        print_trainable_parameters(self.net)

        if HAS_GPU:
            self.net = self.net.cuda()

        self.batch_size = 64
        self.n_epochs = 15
        self.n_workers = 6
        self.n_splits = 10
        self.val_ratio = 0.3
        self.n_tta = 10
        self.lr = 0.00009876
        self.exp_decay_factor = 0.6543
        self.clip_gradients_val = None
        self._write_conf_log("{}".format(self.__dict__))

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
            # To Tensor (float, CxHxW, [0.0, 1.0])  + Normalize
            ToTensor(),
            Normalize(mean_val, std_val)
        ])
        return test_transforms

    def _get_trainval_datasets(self, img_loader, n_splits=5, val_size=0.1, seed=12345, batch_size=32, num_workers=4):

        train_ds = ImageLoaderProxyDataset(img_loader)
        # Resize to 512x512
        train_ds = ResizedDataset(train_ds, (512, 512))
        # Stratified split:        
        sssplit = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
        train_indices, val_indices = next(sssplit.split(img_loader.X_array, img_loader.y_array))

        # Compute class weights and sample weights for training dataset
        train_y_array = img_loader.y_array[train_indices]
        new_indices = over_sample(train_y_array, oversampling_threshold=350)
        new_train_indices = train_indices[new_indices]

        train_sampler = SubsetRandomSampler(new_train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Data augmentations:
        train_transforms = self._get_train_aug()
        test_transforms = self._get_test_aug()
        # Transformed dataset
        data_aug_train_ds = TransformedDataset(train_ds, x_transforms=train_transforms)
        data_aug_val_ds = TransformedDataset(train_ds, x_transforms=test_transforms)

        # Dataloader prefetch + batching
        if HAS_GPU:
            train_batches_ds = OnGPUDataLoader(data_aug_train_ds,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)
            val_batches_ds = OnGPUDataLoader(data_aug_val_ds,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)
            return train_batches_ds, val_batches_ds

        train_batches_ds = DataLoader(data_aug_train_ds,
                                      batch_size=batch_size,
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      drop_last=True)
        val_batches_ds = DataLoader(data_aug_val_ds,
                                    batch_size=batch_size,
                                    sampler=val_sampler,
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

                    # Clip gradients:
                    if self.clip_gradients_val is not None:
                        nn.utils.clip_grad_norm(model.parameters(), 2.0)

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
        msg = "\nOptimizer: %s" % optimizer.__class__.__name__
        msg += "Optimizer parameters: \n"
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

        batch_size = self.batch_size
        n_epochs = self.n_epochs
        num_workers = self.n_workers
        n_splits = self.n_splits
        val_size = self.val_ratio

        lr = self.lr
        optimizer = Adam([{
            'params': self.net.features.parameters(),
            'lr': lr
        }, {
            'params': self.net.classifier.parameters(),
            'lr': 10*lr
        }])

        self._write_conf_log(self._verbose_optimizer(optimizer))

        criterion = nn.CrossEntropyLoss()
        if HAS_GPU:
            criterion = criterion.cuda()
        # lr <- lr_init * gamma ** epoch
        scheduler = ExponentialLR(optimizer, gamma=self.exp_decay_factor)
        onplateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_batches_ds, val_batches_ds = self._get_trainval_datasets(img_loader,
                                                                       seed=SEED,
                                                                       n_splits=n_splits,
                                                                       val_size=val_size,
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
# Data balanced sampling
# *********************************************************************************

def over_sample(targets, oversampling_threshold=350):
    # Oversample training data:
    # - oversample randomly images that count is smaller a threshold

    class_counts = np.zeros((403, ), dtype=np.int)
    for class_index in targets:
        class_counts[class_index] += 1

    classes_to_oversample = np.where(class_counts < oversampling_threshold)[0]

    indices_to_oversample = np.where(np.isin(targets, classes_to_oversample))[0]
    other_indices = np.where(~np.isin(targets, classes_to_oversample))[0]

    rs = RandomOverSampler()
    indices_oversampled, _ = rs.fit_sample(indices_to_oversample[:, None],
                                           targets[indices_to_oversample])
    indices_oversampled = indices_oversampled.ravel()

    new_indices = np.concatenate((other_indices, indices_oversampled))
    return new_indices


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


class RandomCrop(object):

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


class CenterCrop(object):

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
        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = np.pad(img, self.padding, mode='edge')
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i + h, j:j + w, :]


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
