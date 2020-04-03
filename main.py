import tensorflow as tf
import datetime
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import os
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Setup GPU or CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

########################################################################################################################
##########                            Prepare the parser for command line use.                                ##########
########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='CPU')
parser.add_argument("--model", type=str, default="CNN2pooling2fully1_a",
                    help="Available models are: \n CNN1pooling1fully1_a \n CNN2pooling2fully1")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
parser.add_argument("--action", type=str, default="train", help="action should be either train or predict")
parser.add_argument("--checkpoints", type=str, default="logs\\checkpoints",
                    help="Specify relative path to training checkpoints.")
parser.add_argument("--tensorboard", type=str, default="logs\\tb",
                    help="Specify relative path to Tensorboard data. Tensorboard can be invoked using tensorboard --logdir <dir>.")

parser.add_argument("--model_dir", type=str, help="Directory where the selected model is stored.")

args = parser.parse_args()

########################################################################################################################
##########                            Parsed parameters are global variables in this script                   ##########
########################################################################################################################

MODEL = args.model
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
CHECKPOINT_PATH = MODEL + "\\" + args.checkpoints + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '\\'
TENSORBOARD_PATH = MODEL + "\\" + args.tensorboard + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '\\'
LR_SCHEDULE = [(1e-1, 10), (1e-2, 20), (1e-3, 30), (1e-4, 40)]
BASE_LEARNING_RATE = 1e-1
ACTION = args.action
MODEL_DIR = args.model_dir

if args.device == 'GPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*1)])



########################################################################################################################
##########                                          Helper functions                                          ##########
########################################################################################################################


def lr_schedule(epoch, lr_list=LR_SCHEDULE, initial_learning_rate=BASE_LEARNING_RATE):
    '''
    Function which specifies the learning rate schedule.

    Args:
        epoch: Number of epochs used in model training.
        lr_list: List determining the learning rate in terms of executed epochs.
        initial_learning_rate: Learning rate used in the first epoch.

    Returns: learning rate for given epoch.

    '''
    learning_rate = initial_learning_rate
    for mult, start_epoch in lr_list:
        if epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def cut_boxes(tensor, m, n, M, N):
    '''
        TODO: not properly implemented yet.
    Args:
        tensor:
        m:
        n:
        M:
        n:

    Returns:

    '''
    # TODO This implementation needs to be fixed into a tensorflow syntax. The way it is now the
    # sampling is done outside the tf.graph, thus it is not sampled correctly when tf is running.
    m = np.random.randint(0, 28)
    n = np.random.randint(0, 28)
    M = np.random.randint(m + 1, 28)
    N = np.random.randint(n + 1, 28)

    lst = []
    for mm in range(m, M):
        for nn in range(n, N):
            lst.append([mm, nn, 0])

    indices = tf.constant(lst, dtype=tf.int32)
    updates = tf.zeros([(M - m) * (N - n)], dtype=tf.float16)
    updated = tf.tensor_scatter_nd_update(tensor, indices, updates)

    return updated


def data_augmentation(images, labels, flip=True,
                      change_brightness=True,
                      change_contrast=False,
                      cut=False):
    '''
    Function for generating additional augmented training data by horizonal fliping and changing the brightness of the images.

    Args:
        images: (1, 28, 28, 1) features
        labes: labels representing classes 1, 2,...10

    Returns: augmented pair x, y of data.

    '''

    if flip:
        images = tf.image.random_flip_left_right(images)

    if change_brightness:
        images = tf.where(tf.equal(images, 0), images, tf.image.random_brightness(images, 0.2))

    # TODO for some reason tf gives error when change_contrast=True, investigate this!
    if change_contrast:
        images = tf.where(tf.equal(images, 0), images, tf.image.random_contrast(images, lower=0.2, upper=2.0))

    if cut:
        images = cut_boxes(images)

    return images, labels


def data_preprocessing(data):
    '''
    Function prepares data before feeding into a ML classifier.

    Args:
        data: input images of shape (28, 28)

    Returns:
        normalized images of shape (28, 28, 1)
    '''
    images, labels = data
    images = images.astype(np.float16)
    labels = labels.astype(np.float16)

    images = images / 255.0
    images = np.expand_dims(images, axis=3)

    return images, labels


def training_data_preprocessing(data):
    '''

    Args:
        data: input tensor with dimensions (28, 28, 1)

    Returns:
        training set batch sampler fmnist_train_ds
        validation set batch sampler fmnist_validation_ds

    '''

    images, labels = data_preprocessing(data)

    # INFO: The below cutoff is for debugging purposes only.
    # images = images[:60, :, :, :]
    # labels = labels[:60]

    X_train, X_valid, y_train, y_valid = train_test_split(images, labels,
                                                          stratify=labels,
                                                          test_size=0.2,
                                                          random_state=42)

    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    fmnist_train_ds = fmnist_train_ds.map(data_augmentation).shuffle(1000).batch(BATCH_SIZE)

    fmnist_validation_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(data_augmentation).batch(
        BATCH_SIZE)

    return fmnist_train_ds, fmnist_validation_ds


########################################################################################################################
##########              Parent class for the considered models with specified data preprocessing.             ##########
########################################################################################################################


class MyModels(tf.keras.Model):
    '''
        Parent class with useful method implementations that the concrete models share.
    '''

    def __init__(self):
        super(MyModels, self).__init__()

    def train(self, train_df):
        '''

        Args:
            train_df: training data

        Returns: trains the model

        '''

        # Specify training setup
        self.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH + '\\cp-{epoch:04d}.ckpt',
            save_weights_only=True,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1)

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.0001,
            patience=15, verbose=0, mode='auto',
            restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            histogram_freq=1,
            log_dir=TENSORBOARD_PATH)

        train_data, validation_data = training_data_preprocessing(train_df)

        self.fit(train_data,
                 epochs=EPOCHS,
                 callbacks=[earlystop_callback,
                            cp_callback,
                            tensorboard_callback],
                 validation_data=validation_data,
                 validation_freq=1)

    def load_model(self, dir=MODEL_DIR):
        # Loads the weights
        dir = tf.train.latest_checkpoint(dir)
        print(dir)
        self.load_weights(dir)

    def setup(self):
        self.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])


########################################################################################################################
##########                             Specify various DNN architectures.                                     ##########
########################################################################################################################


class CNN2pooling2fully1_a(MyModels):
    def __init__(self):
        super(CNN2pooling2fully1_a, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=4,
                                            strides=(4, 4),
                                            padding="same",
                                            activation=tf.nn.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        # Drop 25% of the input units
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x



class CNN2pooling2fully1_b(MyModels):
    def __init__(self):
        super(CNN2pooling2fully1_b, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=4,
                                            strides=(4, 4),
                                            padding="same",
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 =  tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class CNN2pooling2fully1_c(MyModels):
    def __init__(self):
        super(CNN2pooling2fully1_c, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=[4, 4],
                                            strides=(4, 4),
                                            padding="same",
                                            activation=tf.nn.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class CNN2pooling2fully1_d(MyModels):
    def __init__(self):
        super(CNN2pooling2fully1_d, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(4, 4), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=[4, 4],
                                            strides=(2, 2),
                                            padding="same",
                                            activation=tf.nn.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x




class CNN1pooling1fully1_a(MyModels):
    def __init__(self):
        super(CNN1pooling1fully1_a, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        # Drop 25% of the input units
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x



class CNN1pooling1fully1_b(MyModels):
    def __init__(self):
        super(CNN1pooling1fully1_b, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=4,
                                            strides=(4, 4),
                                            padding="same",
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class CNN1pooling1fully1_c(MyModels):
    def __init__(self):
        super(CNN1pooling1fully1_c, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class CNN1pooling1fully1_d(MyModels):
    def __init__(self):
        super(CNN1pooling1fully1_d, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(4, 4), padding='same',
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x



########################################################################################################################
##########                                    Execution functions.                                            ##########
########################################################################################################################


def main_setup(model=None):
    if model is None:
        if MODEL == 'CNN1pooling1fully1_a':
            model = CNN1pooling1fully1_a()
        elif MODEL == 'CNN1pooling1fully1_b':
            model = CNN1pooling1fully1_b()
        elif MODEL == 'CNN1pooling1fully1_c':
            model = CNN1pooling1fully1_c()
        elif MODEL == 'CNN1pooling1fully1_d':
            model = CNN1pooling1fully1_d()
        elif MODEL == 'CNN2pooling2fully1_a':
            model = CNN2pooling2fully1_a()
        elif MODEL == 'CNN2pooling2fully1_b':
            model = CNN2pooling2fully1_b()
        elif MODEL == 'CNN2pooling2fully1_c':
            model = CNN2pooling2fully1_c()
        elif MODEL == 'CNN2pooling2fully1_d':
            model = CNN2pooling2fully1_d()

    model.setup()
    train_df, test_df = tf.keras.datasets.fashion_mnist.load_data()
    X_test, y_test = data_preprocessing(test_df)
    return model, X_test, y_test, train_df


def main(model=None, action=ACTION):
    model, X_test, y_test, train_df = main_setup(model)

    if action == 'train':
        model.train(train_df)
    elif action == 'predict':
        model.load_model()

    X_train, y_train = data_preprocessing(train_df)
    # Evaluate the model
    loss, acc = model.evaluate(X_train, y_train, verbose=2)
    print("Train data accuracy: {:5.2f}%".format(100 * acc))
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test data accuracy: {:5.2f}%".format(100 * acc))
    print(model.summary())


if __name__ == "__main__":
    main()

