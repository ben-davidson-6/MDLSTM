import tensorflow as tf
import numpy as np
from tensorflow_tools.dataset import Dataset
from tensorflow_tools.network import Network

from layers.two_d_LSTM import MD_parallel


class CircleData(Dataset):
    nx, ny = 256, 256

    @staticmethod
    def _generator(steps):
        # modified from the UNET implementation
        # here https://github.com/jakeret/tf_unet
        nx, ny = CircleData.nx, CircleData.ny
        cnt = 10
        border = 10
        r_min = 5
        r_max = 50
        sigma = 15
        for _ in range(steps):
            image = np.ones((nx, ny, 1))
            label = np.zeros((nx, ny, 2), dtype=np.bool)
            mask = np.zeros((nx, ny), dtype=np.bool)
            for _ in range(cnt):
                a = np.random.randint(border, nx - border)
                b = np.random.randint(border, ny - border)
                r = np.random.randint(r_min, r_max)
                h = np.random.randint(1, 255)

                y, x = np.ogrid[-a:nx - a, -b:ny - b]
                m = x * x + y * y <= r * r
                mask = np.logical_or(mask, m)

                image[m] = h

            label[mask, 1] = 1
            label[~mask, 0] = 1

            image += np.random.normal(scale=sigma, size=image.shape)
            image -= np.amin(image)
            image /= np.amax(image)
            image -= 0.5
            yield image.astype(np.float32), label.astype(np.float32)

    def build_train_dataset(self):
        nx, ny = CircleData.nx, CircleData.ny
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: CircleData._generator(200),
            output_types=(tf.float32, tf.float32),
            output_shapes=([nx, ny, 1], [nx, ny, 2]))
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(1)
        return dataset

    def build_validation_dataset(self):
        nx, ny = CircleData.nx, CircleData.ny
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: CircleData._generator(20),
            output_types=(tf.float32, tf.float32),
            output_shapes=([nx, ny, 1], [nx, ny, 2]))
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(1)
        return dataset

    def build_pipeline(self):
        train_dataset = self.build_train_dataset()
        val_dataset = self.build_validation_dataset()
        self.add_tf_dataset('training', train_dataset)
        self.add_tf_dataset('validation', val_dataset)
        image, label = self.iterator.get_next()
        self.add_gettable_tensor('image', image)
        self.add_gettable_tensor('labels', label)

    def is_training_set(self, dataset_name):
        return 'training' in dataset_name


class MDLSTMNet(Network):

    def __init__(self, units):
        Network.__init__(self)
        self.units = units

    def build_network(self, input_tensors, keep_prob=None, **kwargs):

        if keep_prob is None:
            vals = {'training': 0.5, 'validation': 1.0}
            keep_prob = self.clever_placeholder(tf.float32, [], vals)

        input_tensor = input_tensors['image']
        net = MD_parallel(
            input_tensor,
            self.units,
            cell_computation='leaky',
            initialisation_sigma=0.25)
        out_shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, out_shape[1], out_shape[2], tf.reduce_prod(out_shape[3:])])
        hidden_neurons = 16
        output_neurons = 2
        net = tf.layers.conv2d(net, hidden_neurons, 1, activation=tf.nn.relu)
        net = tf.nn.dropout(net, keep_prob=keep_prob)
        net = tf.layers.conv2d(net, output_neurons, 1, activation=None)
        tf.identity(net, name='logits')
        tf.nn.softmax(net, name='out_softmax')
        tf.argmax(net, axis=tf.rank(net) - 1, name='prediction')
