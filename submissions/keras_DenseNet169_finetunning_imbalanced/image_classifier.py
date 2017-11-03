from __future__ import print_function
import os
from collections import defaultdict
from datetime import datetime

from sklearn.model_selection import StratifiedShuffleSplit

from joblib import delayed
from joblib import Parallel

import numpy as np
import cv2

from skimage.io import imread as skimage_imread

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(sess)

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.losses import categorical_crossentropy

from keras_contrib.applications.densenet import DenseNetImageNet169


SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SIZE = (452, 452)
SEED = 123456789


class ImageClassifier(object):
    def __init__(self):
        self.model = self._build_model()

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

    def fit(self, img_loader):

        batch_size = 64
        valid_ratio = 0.2
        n_epochs = 25

        if 'LOCAL_TESTING' in os.environ:
            print("\n\n------------------------------")
            print("-------- LOCAL TESTING -------")
            print("------------------------------\n\n")
            if 'LOAD_BEST_MODEL' in os.environ:
                load_pretrained_model(self.model, self.logs_path)
                return

        train_gen_builder = BatchGeneratorBuilder(img_loader,
                                                  transform_fn, transform_test_fn,
                                                  chunk_size=batch_size * 10, n_jobs=10)

        gen_train, gen_valid, nb_train, nb_valid, class_weights = \
            train_gen_builder.get_train_valid_generators(batch_size=batch_size,
                                                         valid_ratio=valid_ratio)

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        # Finetunning : all layer after 3rd 'average_pooling2d'
        last_average_pooling_lname = ""
        for l in self.model.layers:
            if 'average_pooling2d' in l.name:
                last_average_pooling_lname = l.name

        index_start = self.model.layers.index(self.model.get_layer(last_average_pooling_lname))
        index_end = len(self.model.layers)
        layer_indices_to_train = list(range(index_start, index_end))
        for index, l in enumerate(self.model.layers):
            l.trainable = False
            if index in layer_indices_to_train:
                l.trainable = True

        self._compile_model(self.model, lr=0.0012345)
        self.model.summary()

        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=n_epochs,
            max_queue_size=batch_size,
            callbacks=get_callbacks(self.model, self.logs_path),
            class_weight=None,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, batch_size) if nb_valid is not None else None,
            verbose=1)

        # Load best trained model:
        load_pretrained_model(self.model, self.logs_path)

    def predict_proba(self, img_loader):

        batch_size = 32
        folder = img_loader.folder
        X_array = img_loader.X_array

        with Parallel(n_jobs=10, backend='threading') as parallel:

            it = _chunk_iterator(parallel, X_array, folder=folder, chunk_size=batch_size*2)
            y_proba = []
            for X in it:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i: i + batch_size]
                    X_batch = parallel(delayed(transform_fn)(x) for x in X_batch)
                    try:
                        X_batch = [x[np.newaxis, :, :, :] for x in X_batch]
                    except IndexError:
                        # single channel
                        X_batch = [
                            x[np.newaxis, np.newaxis, :, :] for x in X_batch]
                    X_batch = np.concatenate(X_batch, axis=0)

                    # 2) Prediction
                    X_aug1 = np.zeros_like(X_batch)
                    X_aug2 = np.zeros_like(X_batch)
                    for i, x in enumerate(X_batch):
                        X_aug1[i, ...] = cv2.flip(x, 0)
                        X_aug2[i, ...] = cv2.flip(x, 1)
                    y_proba0 = self.model.predict(X_batch)
                    y_proba1 = self.model.predict(X_aug1)
                    y_proba2 = self.model.predict(X_aug2)
                    y_proba_batch = 0.33 * (y_proba0 + y_proba1 + y_proba2)
                    y_proba.append(y_proba_batch)
            y_proba = np.concatenate(y_proba, axis=0)
        return y_proba

    def _compile_model(self, model, lr):
        loss = categorical_crossentropy
        model.compile(
            loss=loss, optimizer=Adam(lr=lr),
            metrics=['accuracy', f170])

    def _build_model(self):
        densenet = DenseNetImageNet169(input_shape=SIZE + (3,), include_top=False, weights='imagenet')
        x = densenet.outputs[0]
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.8)(x)
        out = Dense(403, activation='softmax', name='predictions')(x)

        model = Model(densenet.inputs, out)
        model.name = "DenseNet169"
        return model


# ================================================================================================================
# Keras callbacks and metrics
# ================================================================================================================


def step_decay(epoch, model, base=2.0, period=50, verbose=False):
    lr = K.get_value(model.optimizer.lr)
    factor = 1.0 / base if epoch > 0 and epoch % period == 0 else 1.0
    new_lr = lr * factor
    if verbose:
        print("New learning rate: %f" % new_lr)
    return new_lr


def get_callbacks(model, logs_path):
    callbacks = []
    # On plateau reduce LR callback measured on val_loss:
    onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    callbacks.append(onplateau)

    # LR schedule: step decay
    step_decay_f = lambda epoch: step_decay(epoch, model=model, base=1.3, period=3, verbose=True)
    lrscheduler = LearningRateScheduler(step_decay_f)
    callbacks.append(lrscheduler)

    # Store best weights, measured on val_loss
    save_prefix = model.name
    weights_path = os.path.join(logs_path, "weights")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    weights_filename = os.path.join(weights_path, save_prefix + "_best_val_loss.h5")

    model_checkpoint = ModelCheckpoint(weights_filename,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=False)
    callbacks.append(model_checkpoint)

    # Some other callback on local testing
    if 'LOCAL_TESTING' in os.environ:
        from keras.callbacks import TensorBoard, CSVLogger

        csv_logger = CSVLogger(os.path.join(weights_path, 'training_%s.log' % (save_prefix)))
        callbacks.append(csv_logger)

        tboard = TensorBoard('logs', write_grads=True)
        callbacks.append(tboard)

    return callbacks


def false_negatives(y_true, y_pred):
    return K.mean(K.round(K.clip(y_true - y_pred, 0, 1)))


def categorical_crossentropy_with_f1(y_true, y_pred, a=2.0):
    return categorical_crossentropy(y_true, y_pred) + a * (1.0 - K.mean(f1(y_true, y_pred), axis=-1))


def f1(y_true, y_pred):
    # implicit thresholding at 0.5
    y_pred = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    numer = 2.0 * true_positives
    denom = predicted_positives + possible_positives + K.epsilon()
    f1 = numer / denom
    return f1


def f170(y_true, y_pred):
    score = f1(y_true, y_pred)
    score = K.sum(K.round(K.clip((score - 0.7) * 10.0, 0, 1)), axis=0) / K.int_shape(score)[0]
    return score


# ================================================================================================================
# Other useful tools
# ================================================================================================================

def load_pretrained_model(model, logs_path):
    best_weights_filename = os.path.join(logs_path, "weights", "%s_best_val_loss.h5" % model.name)
    print("Load best loss weights: ", best_weights_filename)
    model.load_weights(best_weights_filename)


# ================================================================================================================
# Interface between BatchGeneratorBuilder and ImageLoader
# ================================================================================================================

def imread(filename):
    img = cv2.imread(filename)
    # RGBA -> RGB
    if img is not None:
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = skimage_imread(filename)
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        return img


def _chunk_iterator(parallel, X_array, folder, y_array=None, chunk_size=256):
    for i in range(0, len(X_array), chunk_size):
        X_chunk = X_array[i:i + chunk_size]
        filenames = [os.path.join(folder, '{}'.format(x)) for x in X_chunk]
        X = parallel(delayed(imread)(filename) for filename in filenames)
        if y_array is not None:
            y = y_array[i:i + chunk_size]
            yield X, y
        else:
            yield X


class BatchGeneratorBuilder(object):

    def __init__(self, img_loader,
                 transform_img, transform_test_img,
                 chunk_size, n_jobs):
        self.X_array = img_loader.X_array
        self.y_array = img_loader.y_array

        self.n_classes = img_loader.n_classes
        self.nb_examples = len(img_loader.X_array)
        self.folder = img_loader.folder

        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

        self.transform_img = transform_img
        self.transform_test_img = transform_test_img

        self.shuffle = True

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

            This method is used by the user defined `Classifier` to o build train
            and valid generators that will be used in keras `fit_generator`.

            Parameters
            ==========

            batch_size : int
                size of mini-batches
            valid_ratio : float between 0 and 1
                ratio of validation data

            Returns
            =======

            a 5-tuple (gen_train, gen_valid, nb_train, nb_valid, class_weights) where:
                - gen_train is a generator function for training data
                - gen_valid is a generator function for valid data
                - nb_train is the number of training examples
                - nb_valid is the number of validation examples
                - class_weights
            The number of training and validation data are necessary
            so that we can use the keras method `fit_generator`.
            """

        if valid_ratio > 0.0:
            ssplit = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=SEED)

            train_indices, valid_indices = next(ssplit.split(self.X_array, self.y_array))
            nb_train = len(train_indices)
            nb_valid = len(valid_indices)

            gen_train = self._get_generator(indices=train_indices, batch_size=batch_size)
            gen_valid = self._get_generator(indices=valid_indices, batch_size=batch_size)
        else:
            train_indices = np.arange(self.nb_examples)
            gen_train = self._get_generator(indices=train_indices, batch_size=batch_size)
            nb_train = len(train_indices)
            gen_valid = None
            nb_valid = None

        class_weights = defaultdict(int)
        max_count = 0
        for class_index in self.y_array[train_indices]:
            class_weights[class_index] += 1
            if class_weights[class_index] > max_count:
                max_count = class_weights[class_index]
        for class_index in class_weights:
            class_weights[class_index] = np.log(1.0 + (max_count / class_weights[class_index]) ** 2)

        return gen_train, gen_valid, nb_train, nb_valid, class_weights

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)

        y_stats = defaultdict(int)
        np.random.seed(SEED)

        with Parallel(n_jobs=self.n_jobs, backend='threading') as parallel:
            while True:

                if self.shuffle:
                    np.random.shuffle(indices)
                it = _chunk_iterator(parallel,
                                     X_array=self.X_array[indices], folder=self.folder,
                                     y_array=self.y_array[indices], chunk_size=self.chunk_size)

                for X, y in it:
                    X = parallel(delayed(self.transform_img)(x) for x in X)
                    try:
                        X = [x[np.newaxis, :, :, :] for x in X]
                    except IndexError:
                        # single channel
                        X = [x[np.newaxis, np.newaxis, :, :] for x in X]
                    X = np.concatenate(X, axis=0)
                    X = np.array(X, dtype='float32')

                    for class_index in y:
                        y_stats[class_index] += 1

                    # Convert y to onehot representation
                    y = _to_categorical(y, num_classes=self.n_classes)

                    # 2) Yielding mini-batches
                    for i in range(0, len(X), batch_size):
                        yield X[i:i + batch_size], y[i:i + batch_size]


def _to_categorical(y, num_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    Taken from keras:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    The reason it was taken from keras is to avoid importing theano which
    clashes with pytorch.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_nb_minibatches(nb_samples, batch_size):
    """Compute the number of minibatches for keras.

    See [https://keras.io/models/sequential]
    """
    return (nb_samples // batch_size) + \
           (1 if (nb_samples % batch_size) > 0 else 0)

# ================================================================================================================
# Data augmentations
# ================================================================================================================

from imgaug import augmenters as iaa

from keras_contrib.applications.densenet import preprocess_input

SIZE = (452, 452)  # (w, h)

train_geom_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5, (iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-5, 5),
        order=3,
        mode='edge'
    ))),
])

train_color_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    ])

])

test_geom_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

test_color_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    ])
])


def _transform(x, geom_aug, color_aug):

    # Resize to SIZE
    x = cv2.resize(x, dsize=SIZE, interpolation=cv2.INTER_CUBIC)
    # Data augmentation:
    x = geom_aug.augment_image(x)
    x = color_aug.augment_image(x)

    # to float32
    x = x.astype(np.float32)
    x = preprocess_input(x, data_format='channels_last')
    return x


def transform_fn(x):
    return _transform(x, train_geom_aug, train_color_aug)


def transform_test_fn(x):
    return _transform(x, test_geom_aug, test_color_aug)