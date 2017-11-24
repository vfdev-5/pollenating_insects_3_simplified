from __future__ import division, print_function
import os, sys
from datetime import datetime
from glob import glob
from collections import defaultdict

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
from  torch.nn.functional import sigmoid
from torch.nn import BCEWithLogitsLoss

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn import Module, Sequential, Linear
from torch.autograd import Variable
from torch.optim import Adam

from torchvision.models import squeezenet1_1
from torchvision.transforms import Compose, Normalize, ToTensor

from imblearn.over_sampling import RandomOverSampler


HAS_GPU = torch.cuda.is_available()
SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SIZE = (299, 299)

CONFIG = {
    'nb_train_pairs': 120000,
    'nb_val_pairs': 10000,
    'nb_test_pairs': 10000,

    'batch_size': 64,
    'n_epochs': 15,
    'num_workers': 12,
    'n_splits': 10,
    'val_ratio': 0.10,
    'n_tta': 10,
    'lr': 0.009876,
    'exp_decay_factor': 0.8765,
    'seed': 12345,

    'weight_decay': 0.015
}

print("HAS_GPU: {}".format(HAS_GPU))


# *********************************************************************************
# Networks: SqueezeNet, SqueezeSiameseNet
# *********************************************************************************

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
            nn.Dropout(p=0.65),
            nn.Conv2d(512, 403, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13),
            Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SqueezeSiameseNet(Module):

    def __init__(self):
        super(SqueezeSiameseNet, self).__init__()
        self.net = SqueezeNetPollenatingInsects()

        self.classifier = Sequential(
            Linear(403, 1, bias=False)
        )

    def forward(self, x1, x2):
        x1 = self.net(x1)
        x2 = self.net(x2)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.classifier(x)


# *********************************************************************************
# ImageClassifier
# *********************************************************************************

class ImageClassifier(object):

    def __init__(self):

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.siamese_net = SqueezeSiameseNet()
        print_trainable_parameters(self.siamese_net)

        if HAS_GPU:
            self.siamese_net = self.siamese_net.cuda()

        self.config = CONFIG
        write_conf_log(self.logs_path, "{}".format(self.__dict__))
        self.train_img_loader = None

    def _get_train_aug(self):
        # http://pytorch.org/docs/master/torchvision/models.html
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        train_transforms = Compose([
            RandomCrop(SIZE),
            # Geometry
            RandomChoice([
                RandomAffine(rotation=(-90, 90), scale=(0.85, 1.15), translate=(0.15, 0.15)),
                RandomFlip(proba=0.5, mode='h'),
                RandomFlip(proba=0.5, mode='v'),
            ]),
            # To Tensor (float, CxHxW, [0.0, 1.0]) + Normalize
            ToTensor(),
            ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
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

    def _get_trainval_pairs(self, img_loader):

        trainval_ds = ImageLoaderProxyDataset(img_loader)
        # Resize to 512x512
        trainval_ds = ResizedDataset(trainval_ds, (512, 512))

        # Stratified split:
        sssplit = StratifiedShuffleSplit(n_splits=self.config['n_splits'],
                                         test_size=self.config['val_ratio'],
                                         random_state=self.config['seed'])
        train_indices, val_indices = next(sssplit.split(img_loader.X_array, img_loader.y_array))

        train_class_indices = defaultdict(list)
        val_class_indices = defaultdict(list)

        for i, y in zip(train_indices, img_loader.y_array[train_indices]):
            train_class_indices[y].append(i)

        for _, v in train_class_indices.items():
            assert len(v) > 1

        for i, y in zip(val_indices, img_loader.y_array[val_indices]):
            val_class_indices[y].append(i)

        for k, v in val_class_indices.items():
            if len(v) < 2:
                val_class_indices[k].append(v[0])

        train_pairs = SameOrDifferentPairsDataset(trainval_ds, class_indices=train_class_indices,
                                                  nb_pairs=self.config['nb_train_pairs'], seed=self.config['seed'])
        val_pairs = SameOrDifferentPairsDataset(trainval_ds, class_indices=val_class_indices,
                                                nb_pairs=self.config['nb_val_pairs'], seed=self.config['seed'])

        print("Train pairs: nb_same_pairs=%i | nb_same_pairs_per_class=%i | nb_diff_pairs=%i | nb_samples_per_two_classes=%i" 
              % (len(train_pairs.same_pairs), 
                 train_pairs.nb_same_pairs_per_class, 
                 len(train_pairs.diff_pairs),
                 train_pairs.nb_samples_per_two_classes))

        print("Val pairs: nb_same_pairs=%i | nb_same_pairs_per_class=%i | nb_diff_pairs=%i | nb_samples_per_two_classes=%i" 
              % (len(val_pairs.same_pairs), 
                 val_pairs.nb_same_pairs_per_class, 
                 len(val_pairs.diff_pairs),
                 val_pairs.nb_samples_per_two_classes))
                            
        # Data augmentations:
        train_transforms = self._get_train_aug()
        test_transforms = self._get_test_aug()

        y_transform = lambda y: torch.FloatTensor([y])

        # Transformed pairs
        train_aug_pairs = PairTransformedDataset(train_pairs, x_transforms=train_transforms, y_transforms=y_transform)
        val_aug_pairs = PairTransformedDataset(val_pairs, x_transforms=test_transforms, y_transforms=y_transform)

        # Dataloader prefetch + batching
        _DataLoader = OnGPUDataLoader if HAS_GPU and torch.cuda.is_available() else DataLoader

        train_batches = _DataLoader(train_aug_pairs,
                                    batch_size=self.config['batch_size'],
                                    shuffle=True,
                                    num_workers=self.config['num_workers'],
                                    drop_last=True,
                                    pin_memory=HAS_GPU and torch.cuda.is_available())
        val_batches = _DataLoader(val_aug_pairs,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True,
                                  num_workers=self.config['num_workers'],
                                  drop_last=True,
                                  pin_memory=HAS_GPU and torch.cuda.is_available())
        return train_batches, val_batches

    # def _get_test_datasets(self, img_loader, batch_size=32, num_workers=4):
    #     test_ds = ImageLoaderProxyDataset(img_loader, with_y=False)
    #     # Resize to 512 x 512
    #     test_ds = ResizedDataset(test_ds, (512, 512))
    #     # Data augmentations:
    #     test_transforms = self._get_test_aug()
    #     # Transformed dataset
    #     data_aug_test_ds = TransformedDataset(test_ds, x_transforms=test_transforms)
    #     # Dataloader prefetch + batching
    #     if HAS_GPU:
    #         test_batches_ds = OnGPUDataLoader(data_aug_test_ds,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           num_workers=num_workers,
    #                                           pin_memory=True)
    #         return test_batches_ds
    #
    #     test_batches_ds = DataLoader(data_aug_test_ds,
    #                                  batch_size=batch_size,
    #                                  shuffle=False,
    #                                  num_workers=num_workers)
    #     return test_batches_ds

    def _get_optimizer(self):
        optimizer = Adam([{
            'params': self.siamese_net.net.features.parameters(),
            'lr': self.config['lr_features'],
        }, {
            'params': self.siamese_net.classifier.parameters(),
            'lr': self.config['lr_classifier']
        }],
            weight_decay=self.config['weight_decay']
        )
        return optimizer

    def fit(self, img_loader):

        # store img_loader to setup support set
        self.train_img_loader = img_loader

        # Training - Verification task
        best_model_filenames = glob(os.path.join(self.logs_path, 'model_val_acc*'))
        if len(best_model_filenames) > 0 and os.path.exists(best_model_filenames[0]):
            print("Found weights : %s" % best_model_filenames[0])
            return

        optimizer = self._get_optimizer()
        write_conf_log(self.logs_path, verbose_optimizer(optimizer))

        criterion = BCEWithLogitsLoss()
        if HAS_GPU:
            criterion = criterion.cuda()
        # lr <- lr_init * gamma ** epoch
        scheduler = ExponentialLR(optimizer, gamma=self.config['exp_decay_factor'])
        onplateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_batches, val_batches = self._get_trainval_pairs(img_loader)

        write_csv_log(self.logs_path, "epoch,train_loss,train_prec1,val_loss,val_prec1")

        best_val_acc = 0.0
        n_epochs = self.config['n_epochs']
        for epoch in range(n_epochs):
            scheduler.step()
            # Verbose learning rates:
            print(verbose_optimizer(optimizer))

            # train for one epoch
            ret = train_one_epoch(self.siamese_net, train_batches, criterion, optimizer, epoch, n_epochs,
                                  avg_metrics=[accuracy_logits, ])
            if ret is None:
                break
            train_loss, train_acc = ret

            # evaluate on validation set
            ret = validate(self.siamese_net, val_batches, criterion, avg_metrics=[accuracy_logits,])
            if ret is None:
                break
            val_loss, val_acc = ret

            onplateau_scheduler.step(val_loss)

            # Write a csv log file
            write_csv_log(self.logs_path, "%i,%f,%f,%f,%f" % (epoch, train_loss, train_acc, val_loss, val_acc))

            # remember best accuracy and save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = max(val_acc, best_val_acc)
                save_checkpoint(self.logs_path, 'val_acc', {
                    'epoch': epoch + 1,
                    'state_dict': self.siamese_net.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': optimizer.state_dict()})

    def _setup_support_set_generator(self, img_loader, transforms):
        class_indices = defaultdict(list)
        for i, y in enumerate(img_loader.y_array):
            class_indices[y].append(i)        
        self.support_set_class_indices = class_indices
        
        ds = ImageLoaderProxyDataset(img_loader)
        # Resize to 512x512
        ds = ResizedDataset(ds, (512, 512))
        # Data augmentation
        self.support_set_ds = TransformedDataset(ds, x_transforms=transforms)
        
    def _get_support_set(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        support_set_indices = []        
        for y, indices in self.support_set_class_indices.items():
            index = np.random.randint(len(indices))
            support_set_indices.append(indices[index])
            
        
            
    def predict_proba(self, img_loader):

        # Load pretrained model
        best_model_filenames = glob(os.path.join(self.logs_path, 'model_val_acc*'))
        assert len(best_model_filenames) > 0, "No pretrained models found"
        load_checkpoint(best_model_filenames[0], model=self.siamese_net)

        # switch to evaluate mode
        self.siamese_net.eval()

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
# Dataflow
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


def _create_same_pairs(labels_indices, nb_samples_per_class):
    same_pairs = []
    for indices in labels_indices.values():
        same_pairs.extend([np.random.choice(indices, size=2, replace=False) for _ in range(nb_samples_per_class)])
    return np.array(same_pairs)


def _create_diff_pairs(labels_indices, nb_samples_per_two_classes):
    diff_pairs = []
    for i, indices1 in enumerate(labels_indices.values()):
        for j, indices2 in enumerate(labels_indices.values()):
            if i <= j:
                continue
            ind1 = np.random.choice(indices1, size=nb_samples_per_two_classes)
            ind2 = np.random.choice(indices2, size=nb_samples_per_two_classes)
            diff_pairs.extend([[_i, _j] for _i, _j in zip(ind1, ind2)])
    return np.array(diff_pairs)


class SameOrDifferentPairsDataset(ProxyDataset):
    """
    Create a dataset of pairs uniformly sampled from input dataset
    Pairs are set of two images classified as
        - 'same' if images are from the same class
        - 'different' if images are from different classes
    """

    def __init__(self, ds, nb_pairs, class_indices=None, shuffle=True, seed=None):
        super(SameOrDifferentPairsDataset, self).__init__(ds)
        self.nb_pairs = nb_pairs

        if class_indices is None:
            # get mapping y_label -> indices
            class_indices = defaultdict(list)
            for i, (_, y) in enumerate(ds):
                class_indices[y].append(i)

        if shuffle and seed is not None:
            np.random.seed(seed)                
                
        half_nb_pairs = int(nb_pairs // 2)
        self.nb_same_pairs_per_class = int(np.ceil(half_nb_pairs / len(class_indices)))
        self.same_pairs = _create_same_pairs(class_indices, self.nb_same_pairs_per_class)
        if len(self.same_pairs) > half_nb_pairs:
            if shuffle:
                np.random.shuffle(self.same_pairs)
            self.same_pairs = self.same_pairs[:half_nb_pairs, :]

        self.nb_samples_per_two_classes = int(np.ceil(nb_pairs / (len(class_indices) * (len(class_indices) - 1))))
        self.diff_pairs = _create_diff_pairs(class_indices, self.nb_samples_per_two_classes)
        if len(self.diff_pairs) > half_nb_pairs:
            if shuffle:
                np.random.shuffle(self.diff_pairs)            
            self.diff_pairs = self.diff_pairs[:half_nb_pairs, :]

        self.pairs = np.concatenate((self.same_pairs, self.diff_pairs), axis=0)
        if shuffle:
            np.random.shuffle(self.pairs)

    def __len__(self):
        return self.nb_pairs

    def __getitem__(self, index):
        i1, i2 = self.pairs[index, :]
        x1, y1 = self.ds[i1]
        x2, y2 = self.ds[i2]
        return [x1, x2], int(y1 == y2)


class PairTransformedDataset(TransformedDataset):
    def __getitem__(self, index):
        (x1, x2), y = self.ds[index]
        x1 = self.x_transforms(x1)
        x2 = self.x_transforms(x2)
        if self.y_transforms is not None:
            y = self.y_transforms(y)
        return [x1, x2], y


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

    def _to_cuda(self, t):
        if not t.is_pinned():
            t = t.pin_memory()
        return t.cuda(async=True)

    def __next__(self):
        batch = super(OnGPUDataLoaderIter, self).__next__()
        cuda_batch = []
        for b in batch:  # b is (batch_x, batch_y) or ((batch_x1, batch_x2, ...), (batch_y1, batch_y2, ...))
            if torch.is_tensor(b):
                cuda_batch.append(self._to_cuda(b))
            else:
                assert isinstance(b, tuple) or isinstance(b, list)
                cuda_b = []
                for _b in b:
                    assert torch.is_tensor(_b)
                    cuda_b.append(self._to_cuda(_b))
                cuda_batch.append(cuda_b)
        return cuda_batch

    next = __next__  # Python 2 compatibility


class OnGPUDataLoader(DataLoader):

    def __iter__(self):
        return OnGPUDataLoaderIter(self)


# *********************************************************************************
# ImgAug
# *********************************************************************************

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


# *********************************************************************************
# Training utils
# *********************************************************************************

def train_one_epoch(model, train_batches, criterion, optimizer, epoch, n_epochs, avg_metrics=None):
    """
    :param model: class derived from nn.Module
    :param train_batches: instance of DataLoader
    :param criterion: loss function, callable with signature loss = criterion(batch_y_pred, batch_y)
    :param optimizer:
    :param epoch:
    :param n_epochs:
    :param avg_metrics: list of metrics functions, e.g. [metric_fn1, metric_fn2, ...]
        for example, accuracy(batch_y_pred_tensor, batch_y_true_tensor) -> value
    :return: list of averages, [loss, ] or [loss, metric1, metric2] if metrics is defined
    """

    # Loss
    average_meters = [AverageMeter()]

    if avg_metrics is not None:
        average_meters.extend([AverageMeter() for _ in avg_metrics])

    # switch to train mode
    model.train()
    try:
        with get_tqdm(total=len(train_batches)) as pbar:
            for i, (batch_x, batch_y) in enumerate(train_batches):

                assert torch.is_tensor(batch_y)
                batch_size = batch_y.size(0)

                if isinstance(batch_x, list):
                    batch_x = [Variable(batch_, requires_grad=True) for batch_ in batch_x]
                else:
                    batch_x = [Variable(batch_x, requires_grad=True)]
                batch_y = Variable(batch_y)

                # compute output and measure loss
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                average_meters[0].update(loss.data[0], batch_size)

                prefix_str = "Epoch: {}/{}".format(epoch + 1, n_epochs)
                pbar.set_description_str(prefix_str, refresh=False)
                post_fix_str = "Loss {loss.avg:.4f}".format(loss=average_meters[0])

                # measure metrics
                if avg_metrics is not None:
                    for _fn, av_meter in zip(avg_metrics, average_meters[1:]):
                        v = _fn(batch_y_pred.data, batch_y.data)
                        av_meter.update(v, batch_size)
                        post_fix_str += " | {name} {av_meter.avg:.3f}".format(name=_fn.__name__, av_meter=av_meter)

                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return [m.avg for m in average_meters]
    except KeyboardInterrupt:
        return None


def validate(model, val_batches, criterion, avg_metrics=None, full_data_metrics=None):
    """
    :param model:
    :param val_batches:
    :param criterion:
    :param avg_metrics:
    :param full_data_metrics:
    :return:
    """

    # Loss
    average_meters = [AverageMeter()]

    if avg_metrics is not None:
        average_meters.extend([AverageMeter() for _ in avg_metrics])

    y_true_full = []
    y_pred_full = []

    # switch to evaluate mode
    model.eval()
    try:
        with get_tqdm(total=len(val_batches)) as pbar:
            for i, (batch_x, batch_y) in enumerate(val_batches):

                assert torch.is_tensor(batch_y)
                batch_size = batch_y.size(0)

                if isinstance(batch_x, list):
                    batch_x = [Variable(batch_, volatile=True) for batch_ in batch_x]
                else:
                    batch_x = [Variable(batch_x, volatile=True)]
                batch_y = Variable(batch_y, volatile=True)

                # compute output and measure loss
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                average_meters[0].update(loss.data[0], batch_size)

                if full_data_metrics is not None:
                    _batch_y = batch_y.data
                    if _batch_y.cuda:
                        _batch_y = _batch_y.cpu()
                    y_true_full.append(_batch_y.numpy())
                    _batch_y_pred = batch_y_pred.data
                    if _batch_y_pred.cuda:
                        _batch_y_pred = batch_y_pred.cpu()
                    y_pred_full.append(_batch_y_pred.numpy())

                # measure average metrics
                post_fix_str = "Loss {loss.avg:.4f}".format(loss=average_meters[0])
                # measure metrics
                if avg_metrics is not None:
                    for _fn, av_meter in zip(avg_metrics, average_meters[1:]):
                        v = _fn(batch_y_pred.data, batch_y.data)
                        av_meter.update(v, batch_size)
                        post_fix_str += " | {name} {av_meter.avg:.3f}".format(name=_fn.__name__, av_meter=av_meter)

                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

            if full_data_metrics is not None:
                res = []
                for _fn in full_data_metrics:
                    res.append(_fn(y_true_full, y_pred_full))
                return [m.avg for m in average_meters], res
            else:
                return [m.avg for m in average_meters]
    except KeyboardInterrupt:
        return None


def save_checkpoint(logs_path, val_metric_name, state):
    best_model_filenames = glob(os.path.join(logs_path, 'model_%s*' % val_metric_name))
    for fn in best_model_filenames:
        os.remove(fn)
    best_model_filename = 'model_%s={val_metric_name:.4f}.pth.tar' % val_metric_name
    best_model_filename = best_model_filename.format(
        val_metric_name=state[val_metric_name]
    )
    torch.save(state, os.path.join(logs_path, best_model_filename))


def load_checkpoint(filename, model, optimizer=None):
    print("Load checkpoint: %s" % filename)
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])


def write_csv_log(logs_path, line):
    csv_file = os.path.join(logs_path, 'log.csv')
    _write_log(csv_file, line)


def write_conf_log(logs_path, line):
    conf_file = os.path.join(logs_path, 'conf.log')
    _write_log(conf_file, line)


def _write_log(filename, line):
    d = 'w' if not os.path.exists(filename) else 'a'
    with open(filename, d) as w:
        w.write(line + '\n')


def verbose_optimizer(optimizer):
    msg = "\nOptimizer: %s\n" % optimizer.__class__.__name__
    for pg in optimizer.param_groups:
        msg += "- Param group: \n"
        for k in pg:
            if k == 'params':
                continue
            msg += "\t{}: {}\n".format(k, pg[k])
    return msg


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


def accuracy_logits(y_logits, y_true):
    y_pred = sigmoid(y_logits).data
    return accuracy(y_pred, y_true)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    if output.size(1) > 1:
        _, pred = output.topk(maxk)
    else:
        pred = torch.round(output)

    if len(target.size()) == 1:
        target = target.view(-1, 1)

    if pred.type() != target.type():
        target = target.type_as(pred)

    correct = pred.eq(target.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k * (1.0 / batch_size))
    return res if len(topk) > 1 else res[0]


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
