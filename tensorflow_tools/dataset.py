import tensorflow as tf
import os


class Dataset:
    def __init__(self,):
        self.tf_datasets = []
        self.iterator = None
        self.input_tensors = {}

    def build_pipeline(self, *args):
        """
        Create various tf.data.Dataset's and then add
        them to the Dataset using self.add_tf_dataset(name, dataset)
        """
        raise NotImplementedError('Must build input pipeline')

    def gpu_batch(self, inputs, gpus):
        raise NotImplementedError('Must build split for data parallelism')

    def is_training_set(self, name):
        """
        Return True when name is the name of an
        added dataset which you would like to run a
        TRAIN_STEP on
        """
        raise NotImplementedError('must specify training set')

    def _make_iterator(self, out_t, out_s):
        self.iterator = tf.data.Iterator.from_structure(out_t, out_s)

    def add_gettable_tensor(self, name, t):
        """Add to inputs given to network.build_network"""
        self.input_tensors[name] = t
        tf.identity(t, name)

    def get_inputs(self):
        """What is handed to network.build_network for inputs"""
        assert len(self.input_tensors) > 0, 'Must add_gettable_tensor in input pipeline'
        return self.input_tensors

    def add_tf_dataset(self, name, dataset):
        """
        add datasets to an initialisable iterator, this is so that
        in train.dataset_epoch we can connect different inputs
        """
        # if first dataset create iterator
        if self.iterator is None:
            self._make_iterator(dataset.output_types, dataset.output_shapes)
        self.iterator.make_initializer(dataset, name=name)
        assert name not in self.tf_datasets, 'Name {} already in datasets {}'.format(name, self.tf_datasets)
        self.tf_datasets.append(name)
