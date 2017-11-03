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

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.losses import categorical_crossentropy

from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras_contrib.applications.densenet import DenseNetImageNet161
from keras.utils.data_utils import get_file

from keras_contrib.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input as tf_preprocess_input


SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SEED = 123456789

BASE_WEIGHT_URL = 'https://github.com/vfdev-5/pollenating_insects_3/releases/download/v1.0/'


class ImageClassifier(object):
    def __init__(self):
        self.model = self._build_model()

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

    def fit(self, img_loader):

        batch_size = 32
        valid_ratio = 0.2
        n_epochs = 5

        if 'LOCAL_TESTING' in os.environ:
            print("\n\n------------------------------")
            print("-------- LOCAL TESTING -------")
            print("------------------------------\n\n")
            if 'LOAD_BEST_MODEL' in os.environ:
                load_pretrained_model(self.model, self.logs_path)
                return

        train_gen_builder = BatchGeneratorBuilder(img_loader, chunk_size=batch_size * 10, n_jobs=10)

        gen_train, gen_valid, nb_train, nb_valid, class_weights = \
            train_gen_builder.get_train_valid_generators(batch_size=batch_size,
                                                         valid_ratio=valid_ratio)

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        self._compile_model(self.model, lr=0.01)
        # self.model.summary()

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
        test_gen_builder = BatchGeneratorBuilder(img_loader, chunk_size=batch_size * 10, n_jobs=10)

        test_gen, nb_test = test_gen_builder.get_test_generator(batch_size=batch_size)

        # Perform TTA:
        n = 3
        y_probas = []
        for i in range(n):
            y_probas.append(self.model.predict_generator(test_gen))

        y_probas = np.mean(y_probas)
        return y_probas

    def _compile_model(self, model, lr):
        loss = categorical_crossentropy
        model.compile(
            loss=loss, optimizer=Adam(lr=lr),
            metrics=['accuracy', f170])

    def _build_model(self):

        # Make ensemble of the best pretrained models :
        # - Inception-ResNetV2
        m1 = InceptionResNetV2(input_shape=(451, 451, 3), include_top=False, weights=None)
        x1 = m1.outputs[0]
        # Classification block
        x1 = GlobalAveragePooling2D(name='avg_pool1')(x1)
        out1 = Dense(403, activation='softmax', name='predictions_m1')(x1)

        model1 = Model(m1.inputs, out1)
        model1.name = "InceptionResNetV2-451x451"
        weights_filename = 'InceptionResNetV2_best_val_loss.h5'
        weights_path = get_file(weights_filename,
                                BASE_WEIGHT_URL + weights_filename,
                                cache_subdir='models')
        model1.load_weights(weights_path)
        for l in model1.layers:
            l.trainable = False

        # - DenseNet161
        m2 = DenseNetImageNet161(input_shape=(452, 452, 3), include_top=False, weights=None)
        x2 = m2.outputs[0]
        # Classification block
        x2 = GlobalAveragePooling2D(name='avg_pool2')(x2)
        out2 = Dense(403, activation='softmax', name='predictions_m2')(x2)

        model2 = Model(m2.inputs, out2)
        model2.name = "DenseNet161-452x452"
        weights_filename = 'DenseNet161_best_val_loss.h5'
        weights_path = get_file(weights_filename,
                                BASE_WEIGHT_URL + weights_filename,
                                cache_subdir='models')
        model2.load_weights(weights_path)
        for l in model2.layers:
            l.trainable = False

        # Final classification layer:
        merge_layer = Concatenate()([model1.outputs[0], model2.outputs[0]])
        out = Dense(403, activation='softmax', name='final_predictions')(merge_layer)

        model = Model([model1.inputs[0], model2.inputs[0]], out)
        model.name = "Ensemble_0"
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

        csv_logger = CSVLogger(os.path.join(weights_path, 'training_%s.log' % save_prefix))
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

def _imread(filename):
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


def _imread_transform(filename, fn):
    img = _imread(filename)
    return fn(img)


def _transform_fn_451(x):
    return _transform_fn(x, size=(451, 451), preprocess_input_fn=tf_preprocess_input)


def _transform_fn_452(x):
    return _transform_fn(x, size=(452, 452), preprocess_input_fn=densenet_preprocess_input)


def _transform_test_fn_451(x):
    return _transform_test_fn(x, size=(451, 451), preprocess_input_fn=tf_preprocess_input)


def _transform_test_fn_452(x):
    return _transform_test_fn(x, size=(452, 452), preprocess_input_fn=densenet_preprocess_input)


class BatchGeneratorBuilder(object):

    def __init__(self, img_loader, transform_fn, transform_test_fn, chunk_size, n_jobs):
        self.X_array = img_loader.X_array
        self.y_array = img_loader.y_array

        self.n_classes = img_loader.n_classes
        self.nb_examples = len(img_loader.X_array)
        self.folder = img_loader.folder

        self.transform_fn = transform_fn
        self.transform_test_fn = transform_test_fn

        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

        self.shuffle = True

    def get_train_valid_generators(self, batch_size=32, valid_ratio=0.1):
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

            gen_train = self._get_generator(indices=train_indices,
                                            batch_size=batch_size,
                                            transform_fn=[self.transform_fn, ])
            gen_valid = self._get_generator(indices=valid_indices,
                                            batch_size=batch_size,
                                            transform_fn=[self.transform_test_fn, ])
        else:
            train_indices = np.arange(self.nb_examples)
            gen_train = self._get_generator(indices=train_indices,
                                            batch_size=batch_size,
                                            transform_fn=[self.transform_fn, ])
            nb_train = len(train_indices)
            gen_valid = None
            nb_valid = None

        class_weights = None

        return gen_train, gen_valid, nb_train, nb_valid, class_weights

    def get_test_generator(self, batch_size=32):
        test_indices = np.arange(self.nb_examples)
        gen_test = self._get_generator(indices=test_indices,
                                       batch_size=batch_size,
                                       transform_fn=[self.transform_test_fn, ])
        nb_test = len(test_indices)
        return gen_test, nb_test

    def _get_generator(self, transform_fn, indices=None, batch_size=32):
        if indices is None:
            indices = np.arange(self.nb_examples)

        assert transform_fn is not None

        assert isinstance(transform_fn, list)
        if isinstance(transform_fn, list):
            for fn in transform_fn:
                assert callable(fn)

        y_stats = defaultdict(int)
        np.random.seed(SEED)

        with Parallel(n_jobs=self.n_jobs, backend='threading') as parallel:
            while True:

                if self.shuffle:
                    np.random.shuffle(indices)

                X_array = self.X_array[indices]
                chunk_size = self.chunk_size
                folder = self.folder
                y_array = self.y_array[indices]

                for i in range(0, len(X_array), chunk_size):
                    X_chunk = X_array[i:i + chunk_size]

                    if len(X_chunk) < chunk_size:
                        continue

                    filenames = [os.path.join(folder, '{}'.format(x)) for x in X_chunk]
                    X = []
                    for fn in transform_fn:
                        res = parallel(delayed(_imread_transform)(filename, fn) for filename in filenames)
                        X.append(np.array(res))

                    if y_array is not None:
                        y = y_array[i:i + chunk_size]
                        for class_index in y:
                            y_stats[class_index] += 1
                        # Convert y to onehot representation
                        y = _to_categorical(y, num_classes=self.n_classes)
                    else:
                        y = None

                    # 2) Yielding mini-batches
                    ll = len(X)
                    for j in range(0, chunk_size, batch_size):
                        batch_x = [X[k][j:j + batch_size] for k in range(ll)] if ll > 1 else X[0][j:j + batch_size]
                        if y is not None:
                            yield batch_x, y[j:j + batch_size]
                        else:
                            yield batch_x


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


def _transform(x, size, preprocess_input_fn, geom_aug=None, color_aug=None):

    # Resize to SIZE
    x = cv2.resize(x, dsize=size[::-1], interpolation=cv2.INTER_CUBIC)
    # Data augmentation:
    if geom_aug is not None:
        x = geom_aug.augment_image(x)
    if color_aug is not None:
        x = color_aug.augment_image(x)

    # to float32
    x = x.astype(np.float32)
    x = preprocess_input_fn(x)
    return x


def _transform_fn(x, size, preprocess_input_fn):
    return _transform(x, size, preprocess_input_fn, train_geom_aug, None)


def _transform_test_fn(x, size, preprocess_input_fn):
    return _transform(x, size, preprocess_input_fn, test_geom_aug, test_color_aug)
