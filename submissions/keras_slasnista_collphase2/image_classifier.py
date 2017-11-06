import numpy as np

from skimage.transform import resize
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from rampwf.workflows.image_classifier import get_nb_minibatches

from skimage.transform import rotate


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class ImageClassifier(object):
    def __init__(self, batch_size=16):

        self.batch_size = batch_size
        self.model1 = self._build_model(base="xception")
        self.model2 = self._build_model(base="inception_v3")

    def _build_model(self, base="xception"):
        inp = Input((299, 299, 3))

        if base == "xception":
            base_model = Xception(
                include_top=True, weights='imagenet',
                input_tensor=inp)

        elif base == "inception_v3":
            base_model = InceptionV3(
                include_top=True, weights='imagenet',
                input_tensor=inp
            )

        out = Dense(
            403, activation='softmax',
            name='classifier')(base_model.get_layer("avg_pool").output)

        model = Model(inp, out)

        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])

        return model

    def _transform(self, x, dim=299):

        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        h, w = x.shape[:2]
        min_shape = min(h, w)
        x = x[h // 2 - min_shape // 4:h // 2 + min_shape // 4,
            w // 2 - min_shape // 4:w // 2 + min_shape // 4]

        # random rotation
        if np.random.rand() < 0.5:
            x = rotate(x, angle=np.random.randint(180), preserve_range=True)

        x = resize(x, (dim, dim), preserve_range=True)

        # horizontal flip
        if np.random.rand() < 0.5:
            x = flip_axis(x, axis=0)

        # vertical flip
        if np.random.rand() < 0.5:
            x = flip_axis(x, axis=1)

        x /= 255.
        x -= 0.5
        x *= 2.

        return x

    def _transform_test(self, x, dim=299):
        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        h, w = x.shape[:2]
        min_shape = min(h, w)
        x = x[h // 2 - min_shape // 4:h // 2 + min_shape // 4,
            w // 2 - min_shape // 4:w // 2 + min_shape // 4]
        x = resize(x, (dim, dim), preserve_range=True)

        x /= 255.
        x -= 0.5
        x *= 2.

        return x

    def _build_train_generator(self, img_loader, indices, batch_size,
                               shuffle=False):
        indices = indices.copy()
        nb = len(indices)
        X = np.zeros((batch_size, 299, 299, 3))
        Y = np.zeros((batch_size, 403))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                Y[:] = 0
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    x = self._transform(x)
                    X[i] = x
                    Y[i, y] = 1
                yield X[:bs], Y[:bs]

    def _build_test_generator(self, img_loader, batch_size):
        nb = len(img_loader)
        X = np.zeros((batch_size, 299, 299, 3))
        while True:
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                for i, img_index in enumerate(range(start, stop)):
                    x = img_loader.load(img_index)
                    x = self._transform_test(x)
                    X[i] = x
                yield X[:bs]

    def fit(self, img_loader):
        np.random.seed(24)
        nb = len(img_loader)
        nb_train = int(nb * 1.)
        # nb_valid = nb - nb_train
        indices = np.arange(nb)
        np.random.shuffle(indices)
        ind_train = indices[0: nb_train]
        # ind_valid = indices[nb_train:]

        gen_train = self._build_train_generator(
            img_loader,
            indices=ind_train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # no validation to have more data for training

        # gen_valid = self._build_train_generator(
        #     img_loader,
        #     indices=ind_valid,
        #     batch_size=self.batch_size,
        #     shuffle=True
        # )

        batch_size = 16
        epochs = 4

        # training model 1
        self.model1.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=epochs,
            max_queue_size=64,
            workers=1,
            use_multiprocessing=True,
            verbose=1)

        self.model1.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])

        self.model1.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=1,
            max_queue_size=64,
            workers=1,
            use_multiprocessing=True,
            verbose=1)

        # training model 2
        self.model2.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=epochs,
            max_queue_size=64,
            workers=1,
            use_multiprocessing=True,
            verbose=1)

        self.model2.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])

        self.model2.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=epochs,
            max_queue_size=64,
            workers=1,
            use_multiprocessing=True,
            verbose=1)

    def predict_proba(self, img_loader):
        nb_test = len(img_loader)
        gen_test = self._build_test_generator(img_loader, self.batch_size)

        # predict with model1
        y_pred1 = self.model1.predict_generator(
            gen_test,
            steps=get_nb_minibatches(nb_test, self.batch_size),
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            verbose=0
        )

        # predict with model2
        y_pred2 = self.model2.predict_generator(
            gen_test,
            steps=get_nb_minibatches(nb_test, self.batch_size),
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            verbose=0
        )

        y_pred1 = np.expand_dims(y_pred1, axis=0)
        y_pred2 = np.expand_dims(y_pred2, axis=0)

        y_pred = np.concatenate([y_pred1, y_pred2])
        y_pred = np.mean(y_pred, axis=0)

        return y_pred