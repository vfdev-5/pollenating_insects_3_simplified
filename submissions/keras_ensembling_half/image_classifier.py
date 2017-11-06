from __future__ import print_function

import os
from datetime import datetime

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from joblib import delayed
from joblib import Parallel

import numpy as np
import cv2

from skimage.io import imread as skimage_imread

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.losses import categorical_crossentropy

from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as irv2_preprocess_input
from keras_contrib.applications.densenet import DenseNetImageNet161, preprocess_input as densenet_preprocess_input


SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))

SEED = 123456789
N_JOBS = 10
SIZE = (224, 224)  # Minimal size for ResNet50


class ImageClassifier(object):
    def __init__(self):
        self._build_models()

        self.batch_size = 16
        self.valid_ratio = 0.2
        self.n_epochs = 6
        self.n_tta = 10
        self.learning_rates = [
            0.000234,  # inception-resnet
            0.000123,  # resnet
            0.000234,  # xception
            0.000015,  # densenet
        ]

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

    def _train_first_level(self, img_loader):
        print("\n\n First level training \n")
        for i, (model, preproc_fns) in enumerate(zip(self.models, self.preprocessing_fns)):

            print("\n---------------------------------------------------------------")
            print("\tTrain model : %s (%i / %i)" % (model.name, i + 1, len(self.models)))
            print("---------------------------------------------------------------\n")

            best_weights_filename = os.path.join(self.logs_path, "weights", "%s_best_val_loss.h5" % model.name)
            if os.path.exists(best_weights_filename):
                print("Found weights at %s" % best_weights_filename)
                continue

            train_gen_builder = BatchGeneratorBuilder(img_loader,
                                                      shuffle=True,
                                                      transform_img=preproc_fns[0],
                                                      transform_test_img=preproc_fns[1],
                                                      chunk_size=self.batch_size * 2,
                                                      n_jobs=N_JOBS)

            gen_train, gen_valid, nb_train, nb_valid, class_weights = \
                train_gen_builder.get_train_valid_generators(batch_size=self.batch_size,
                                                             valid_ratio=self.valid_ratio)

            print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

            self._compile_model(model, lr=self.learning_rates[i])

            model.fit_generator(
                gen_train,
                steps_per_epoch=get_nb_minibatches(nb_train, self.batch_size),
                epochs=self.n_epochs,
                max_queue_size=self.batch_size * 2,
                callbacks=get_callbacks(model, self.logs_path),
                class_weight=None,
                validation_data=gen_valid,
                validation_steps=get_nb_minibatches(nb_valid, self.batch_size) if nb_valid is not None else None,
                verbose=1)

    def _predict_first_level(self, img_loader):
        y_probas_1lvl = np.zeros((len(img_loader), img_loader.n_classes * len(self.models)))

        # Load best trained model:
        for i, (model, preproc_fns) in enumerate(zip(self.models, self.preprocessing_fns)):
            load_pretrained_model(model, self.logs_path)

            test_gen_builder = BatchGeneratorBuilder(img_loader,
                                                     shuffle=False,
                                                     transform_img=preproc_fns[0],
                                                     transform_test_img=preproc_fns[1],
                                                     chunk_size=self.batch_size * 2,
                                                     n_jobs=N_JOBS)

            # Perform TTA:
            y_proba = np.zeros((self.n_tta, test_gen_builder.nb_examples, img_loader.n_classes))
            for j in range(self.n_tta):
                print("- TTA round: %i / %i" % (j + 1, self.n_tta))
                test_gen, nb_test = test_gen_builder.get_test_generator(batch_size=self.batch_size)
                y_proba[j, :, :] = model.predict_generator(test_gen,
                                                           steps=get_nb_minibatches(nb_test, self.batch_size),
                                                           max_queue_size=self.batch_size * 2,
                                                           verbose=1)
            y_proba = np.mean(y_proba, axis=0)
            y_probas_1lvl[:, 403 * i:(i + 1) * 403] = y_proba

        return y_probas_1lvl

    def _compute_topk(self, y_probas_1lvl, k):
        y_topk_preds = np.zeros((len(y_probas_1lvl), len(self.models) * k))
        for i in range(len(self.models)):
            y_topk_preds[:, i * k:(i + 1) * k] = np.argsort(y_probas_1lvl[:, 403 * i:(i + 1) * 403],
                                                            axis=1)[:, ::-1][:, :k]
        return y_topk_preds.astype(np.int)

    def fit(self, img_loader):
        self._train_first_level(img_loader)

        first_level_probas_filename = os.path.join(self.logs_path, "y_probas_1lvl.npz")
        if not os.path.exists(first_level_probas_filename):
            y_probas_1lvl = self._predict_first_level(img_loader)
            # Save 1st level probas:
            np.savez_compressed(first_level_probas_filename, y_probas_1lvl=y_probas_1lvl)
        else:
            print("Load y_probas_1lvl from %s" % first_level_probas_filename)
            y_probas_1lvl = np.load(first_level_probas_filename)['y_probas_1lvl']

        # First level mean f_beta score:
        y_true = img_loader.y_array
        y_probas_1lvl_mean = np.mean(y_probas_1lvl.reshape((len(y_probas_1lvl),
                                                            len(self.models),
                                                            img_loader.n_classes)), axis=1)
        res = accuracy_score(y_true, np.argmax(y_probas_1lvl_mean, axis=1))
        print("\nAccuracy from mean first level predictons: {}".format(res))

    def predict_proba(self, img_loader):
        y_probas_1lvl = self._predict_first_level(img_loader)
        y_probas_1lvl_mean = np.mean(y_probas_1lvl.reshape((len(y_probas_1lvl),
                                                            len(self.models),
                                                            img_loader.n_classes)), axis=1)
        return y_probas_1lvl_mean

    def _compile_model(self, model, lr):
        loss = categorical_crossentropy
        model.compile(
            loss=loss,
            optimizer=Adam(lr=lr),
            metrics=['accuracy', f170])

    def _build_models(self):

        pretrained_models = [
            InceptionResNetV2(input_shape=SIZE + (3, ), include_top=False, weights='imagenet'),
            ResNet50(input_shape=SIZE + (3, ), include_top=False, weights='imagenet'),
            Xception(input_shape=SIZE + (3, ), include_top=False, weights='imagenet'),
            DenseNetImageNet161(input_shape=SIZE + (3, ), include_top=False, weights='imagenet'),
        ]

        self.preprocessing_fns = [
            (_transform_fn_irv2, _transform_test_fn_irv2),
            (_transform_fn_resnet, _transform_test_fn_resnet),
            (_transform_fn_xception, _transform_test_fn_xception),
            (_transform_fn_densenet, _transform_test_fn_densenet),
        ]

        self.models = []
        for model in pretrained_models:
            x = model.outputs[0]
            # Classification block
            x = GlobalAveragePooling2D(name='avg_pool_%s' % model.name)(x)
            out = Dense(403, activation='softmax', name='predictions_%s' % model.name)(x)
            new_model = Model(model.inputs[0], out)
            new_model.name = model.name + "_PI3"
            self.models.append(new_model)


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
    step_decay_f = lambda epoch: step_decay(epoch, model=model, base=1.25, period=1, verbose=True)
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
        from keras.callbacks import CSVLogger

        csv_logger = CSVLogger(os.path.join(weights_path, 'training_%s.log' % save_prefix))
        callbacks.append(csv_logger)

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
    assert os.path.exists(best_weights_filename)
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


class BatchGeneratorBuilder(object):

    def __init__(self, img_loader,
                 transform_img,
                 transform_test_img,
                 shuffle,
                 chunk_size,
                 n_jobs):

        self.X_array = img_loader.X_array
        self.nb_examples = len(self.X_array)
        self.folder = img_loader.folder

        self.y_array = img_loader.y_array
        self.n_classes = img_loader.n_classes

        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

        self.transform_img = transform_img
        self.transform_test_img = transform_test_img

        self.shuffle = shuffle

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
        assert self.y_array is not None

        if valid_ratio > 0.0:
            ssplit = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=SEED)

            train_indices, valid_indices = next(ssplit.split(self.X_array, self.y_array))
            nb_train = len(train_indices)
            nb_valid = len(valid_indices)

            gen_train = self._get_generator(self.transform_img, indices=train_indices, batch_size=batch_size)
            gen_valid = self._get_generator(self.transform_test_img, indices=valid_indices, batch_size=batch_size)
        else:
            train_indices = np.arange(self.nb_examples)
            gen_train = self._get_generator(self.transform_img, indices=train_indices, batch_size=batch_size)
            nb_train = len(train_indices)
            gen_valid = None
            nb_valid = None

        class_weights = None
        # class_weights = defaultdict(int)
        # max_count = 0
        # for class_index in self.y_array[train_indices]:
        #     class_weights[class_index] += 1
        #     if class_weights[class_index] > max_count:
        #         max_count = class_weights[class_index]
        # for class_index in class_weights:
        #     class_weights[class_index] = np.log(1.0 + (max_count / class_weights[class_index]) ** 2)

        return gen_train, gen_valid, nb_train, nb_valid, class_weights

    def get_test_generator(self, batch_size=32):
        test_indices = np.arange(self.nb_examples)
        gen_test = self._get_generator(transform_fn=self.transform_test_img,
                                       indices=test_indices,
                                       is_infinite=False,
                                       batch_size=batch_size)
        nb_test = len(test_indices)
        return gen_test, nb_test

    def _get_generator(self, transform_fn, indices, is_infinite=True, batch_size=32):

        assert transform_fn is not None
        assert indices is not None

        np.random.seed(SEED)
        chunk_size = self.chunk_size
        folder = self.folder

        with Parallel(n_jobs=self.n_jobs, backend='threading') as parallel:
            while True:

                if self.shuffle:
                    np.random.shuffle(indices)

                X_array = self.X_array[indices]
                y_array = self.y_array[indices] if self.y_array is not None else None

                for i in range(0, len(X_array), chunk_size):
                    X_chunk = X_array[i:i + chunk_size]

                    if len(X_chunk) < chunk_size and is_infinite:
                        continue

                    filenames = [os.path.join(folder, '{}'.format(x)) for x in X_chunk]

                    X = np.array(parallel(delayed(_imread_transform)(filename, transform_fn) for filename in filenames))

                    if y_array is not None:
                        y = y_array[i:i + chunk_size]
                        # Convert y to onehot representation
                        y = _to_categorical(y, num_classes=self.n_classes)
                    else:
                        y = None

                    # 2) Yielding mini-batches
                    for j in range(0, chunk_size, batch_size):
                        if y is not None:
                            yield X[j:j + batch_size], y[j:j + batch_size]
                        else:
                            yield X[j:j + batch_size]

                # Exit if test mode
                if not is_infinite:
                    return


def _to_categorical(y, num_classes=403):
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
        scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
        translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
        rotate=(-45, 45),
        shear=(-2, 2),
        order=3,
        mode='edge'
    ))),
])

train_color_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
    ]),
])

test_geom_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Affine(
            translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
            rotate=(-45, 45),
            order=3,
            mode='edge'
        ),
    ])

])

test_color_aug = None


def _transform(x, size, preprocess_input_fn, geom_aug=None, color_aug=None):

    # Resize to SIZE
    x = cv2.resize(x, dsize=(size[1]*2, size[0]*2), interpolation=cv2.INTER_NEAREST)
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
    return _transform(x, size, preprocess_input_fn, train_geom_aug, train_color_aug)


def _transform_test_fn(x, size, preprocess_input_fn):
    return _transform(x, size, preprocess_input_fn, test_geom_aug, test_color_aug)


def _transform_fn_irv2(x):
    return _transform_fn(x, size=SIZE, preprocess_input_fn=irv2_preprocess_input)


def _transform_fn_resnet(x):
    return _transform_fn(x, size=SIZE, preprocess_input_fn=resnet_preprocess_input)


def _transform_fn_xception(x):
    return _transform_fn(x, size=SIZE, preprocess_input_fn=xception_preprocess_input)


def _transform_fn_densenet(x):
    return _transform_fn(x, size=SIZE, preprocess_input_fn=densenet_preprocess_input)


def _transform_test_fn_irv2(x):
    return _transform_test_fn(x, size=SIZE, preprocess_input_fn=irv2_preprocess_input)


def _transform_test_fn_resnet(x):
    return _transform_test_fn(x, size=SIZE, preprocess_input_fn=resnet_preprocess_input)


def _transform_test_fn_xception(x):
    return _transform_test_fn(x, size=SIZE, preprocess_input_fn=xception_preprocess_input)


def _transform_test_fn_densenet(x):
    return _transform_test_fn(x, size=SIZE, preprocess_input_fn=densenet_preprocess_input)
