import random

import PIL.Image as im
import PIL.ImageEnhance as ie
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

##############################################################################

def _transform(x):
    img = Image.fromarray(x).convert(input_space)
    img = tf(img)
    return img.numpy()


class TrainDataset(Dataset):
    def __init__(self, img_loader, transform=_transform):
        self.img_loader = img_loader
        self.transform = transform
        self.len = len(self.img_loader)

    def __getitem__(self, index):
        img, label = self.img_loader.load(index)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, img_loader, transform=_transform):
        self.img_loader = img_loader
        self.transform = transform
        self.len = len(self.img_loader)

    def __getitem__(self, index):
        img = self.img_loader.load(index)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.len


class ImageClassifier(object):
    def __init__(self):
        self.net = inceptionv4(num_classes=1000, pretrained='imagenet')
        self.net.classif = nn.Linear(1536, 403)
        self.net.cuda()

        self.settings = {
                'name': 'default',
                'batch_size': 32,
                'nb_epochs': 3,
                'lr': 0.0045,
                'lr_decay': 0.5,
                'aug_params': [True, True, 0, 0, 0, 0],
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'shuffle': False,
                'tta_factor': 15,
                'test_batch_size': 64,
                'pretrained': False,
                'model_url': ''
            }

        """
            pretrained = True is only for local testing purpose (testing augmentation strategies for example).
            It cannot be used on RAMP.
            Currently the model is trained on all images, so it currently doesn't make much sense, I'll provide
            a model trained on a subset.
        """
        self._make_settings('much_too_slow', pretrained=False)

        self.transforms = Transform(self.settings['input_space'],
                                    self.settings['mean'],
                                    self.settings['std'],
                                    self.settings['aug_params'])

        if self.settings['pretrained']:
            self.net.load_state_dict(model_zoo.load_url(self.settings['model_url']))
            self.net.eval()

    def fit(self, img_loader):

        if self.settings['pretrained']:
            return

        batch_size = self.settings['batch_size']
        nb_epochs = self.settings['nb_epochs']
        lr = self.settings['lr']
        lr_decay = self.settings['lr_decay']

        train_set = TrainDataset(img_loader, transform=self.transforms)
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=self.settings['shuffle'],
                                  num_workers=8, pin_memory=True)

        optimizer = optim.RMSprop([{'params': self.net.features.parameters(), 'lr': lr},
                                   {'params': self.net.classif.parameters(), 'lr': 10 * lr}], eps=1.0, momentum=0.9)

        scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: lr_decay ** epoch, lambda epoch: lr_decay ** epoch])

        criterion = nn.CrossEntropyLoss().cuda()

        self.net.train()
        for epoch in tqdm(range(nb_epochs), desc="Epochs", unit="epoch"):
            scheduler.step()

            for X, y in tqdm(train_loader, desc="Mini-batches", unit="batch"):
                X = Variable(X.cuda(async=True))
                y = Variable(y.cuda(async=True))

                optimizer.zero_grad()

                y_pred = self.net(X)

                loss = criterion(y_pred, y)

                loss.backward()

                optimizer.step()

        torch.save(self.net.state_dict(), self.settings['name'] + ".pth")
        # Put the Net in eval mode, only affects batchnorm
        self.net.eval()

    def _get_acc(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
        y_true = y_true.cpu().data.numpy()
        return (y_pred == y_true)

    def predict_proba(self, img_loader):
        tta_factor = self.settings['tta_factor']
        test_batch_size = self.settings['test_batch_size']

        test_set = TestDataset(img_loader, transform=self.transforms)
        test_loader = DataLoader(test_set,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 num_workers=8, pin_memory=True)

        nb = len(img_loader)

        Y = torch.zeros((tta_factor, nb, 403)).cuda()

        for i in tqdm(range(tta_factor), desc="Random Patches", unit="patch"):
            start = 0
            for X in tqdm(test_loader, desc="Batches", unit="batch"):
                X = Variable(X.cuda(async=True), volatile=True)

                y = F.softmax(self.net(X)).data

                bs = y.shape[0]
                stop = start + bs
                Y[i, start:stop, :] = y
                start = stop

        return torch.mean(Y, dim=0).cpu().numpy()

    def _make_settings(self, submission=None, pretrained=None):
        """
            pretrained = True is only for local testing purpose (testing augmentation strategies for example).
            It cannot be used on RAMP.
            Currently the model is trained on all images, so it currently doesn't make much sense, I'll provide
            a model trained on a subset.
        """
        settings = {
            'gotta_go_fast': {
                'name': 'gotta_go_fast',
                'batch_size': 32,
                'nb_epochs': 3,
                'lr': 0.0045,
                'lr_decay': 0.5,
                'aug_params': [True, True, 0, 0, 0, 0],
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'shuffle': False,
                'tta_factor': 15,
                'test_batch_size': 64,
                'pretrained': True,
                'model_url': "https://gitlab.com/aussetg/pollenating_insects_3_simplified/raw/master/pretrained/gotta_go_fast-4c2a9900.pth"
            },
            'much_too_slow': {
                'name': 'much_too_slow',
                'batch_size': 32,
                'nb_epochs': 15,
                'lr': 0.0045,
                'lr_decay': 0.7,
                'aug_params': [True, True, 0.4, 0.4, 0.4, 0.4],
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'shuffle': False,
                'tta_factor': 30,
                'test_batch_size': 64,
                'pretrained': True,
                'model_url': 'https://gitlab.com/aussetg/pollenating_insects_3_simplified/raw/master/pretrained/much_too_slow-7899b4f6.pth'
            },
            'less_slow': {
                'name': 'less_slow',
                'batch_size': 32,
                'nb_epochs': 12,
                'lr': 0.0045,
                'lr_decay': 0.7,
                'aug_params': [True, True, 0.4, 0.4, 0.4, 0.4],
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'shuffle': False,
                'tta_factor': 30,
                'test_batch_size': 64,
                'pretrained': True,
                'model_url': 'https://gitlab.com/aussetg/pollenating_insects_3_simplified/raw/master/pretrained/less_slow-16dc25be.pth'
            }
        }
        if submission is not None:
            self.settings = settings[submission]
        if pretrained is not None:
            self.settings['pretrained'] = False


def _make_variable(X, pinned=False, volatile=False):
    return Variable(torch.from_numpy(X).pin_memory(), volatile=volatile).cuda(async=True)


def _make_variable_test(X, pinned=False, volatile=False):
    return Variable(torch.from_numpy(X).contiguous().pin_memory(), volatile=volatile).cuda(async=True)


def _flatten(x):
    return x.view(x.size(0), -1)


######################### Preprocessing #########################

input_space = 'RGB'
input_size = [3, 299, 299]
input_range = [0, 1]
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class RandomFlip(object):
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }

        return dispatcher[random.randint(0, 3)]  # randint is inclusive


class RandomRotate(object):
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }

        return dispatcher[random.randint(0, 5)]  # randint is inclusive


class PILColorBalance(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)


class PILContrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)


class PILSharpness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)


class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                 flip=True,
                 colorbalance=0.4,
                 contrast=0.4,
                 brightness=0.4,
                 sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))

class Transform(object):
    def __init__(self, input_space, mean, std, params):
        self.params = params
        self.input_space = input_space
        self.mean = mean
        self.std = std
        self.tf = transforms.Compose([
                transforms.Scale(340),
                transforms.RandomCrop(299),
                PowerPIL(*params),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
                ])

    def __call__(self, img):
        img = Image.fromarray(img).convert(self.input_space)
        img = self.tf(img)
        return img.numpy()


tf = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Scale(340),
    transforms.RandomCrop(299),
    PowerPIL(),
    #PowerPIL(True, True, 0, 0, 0, 0),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

tf_test = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale(340),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


################################ Inception V4 ################################

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            nn.AvgPool2d(8, count_include_pad=False)
        )
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


def inceptionv4(num_classes=1001, pretrained='imagenet'):
    pretrained_settings = {
        'inceptionv4': {
            'imagenet': {
                'url': 'http://webia.lip6.fr/~cadene/Downloads/inceptionv4-97ef9c30.pth',
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'num_classes': 1000
            },
            'imagenet+background': {
                'url': 'http://webia.lip6.fr/~cadene/Downloads/inceptionv4-97ef9c30.pth',
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'num_classes': 1001
            }
        }
    }
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_classif = nn.Linear(1536, 1000)
            new_classif.weight.data = model.classif.weight.data[1:]
            new_classif.bias.data = model.classif.bias.data[1:]
            model.classif = new_classif

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)
    return model
