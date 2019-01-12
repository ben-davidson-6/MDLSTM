import tensorflow as tf


class Network:

    def __init__(self):
        self.feeds = {}

    def build_network(self, inputs, **kwargs):
        """
        inputs -> dataset.get_inputs(): dictionary

        Actually define the network, doesnt return anything
        as only expected to construct graph. The tensors required
        for whatever purpose should be named and gotten through
        graph.get_tensor_by_name wherever appropriate
        """
        raise NotImplementedError('Must create a builder function')

    def clever_placeholder(self, dtype, shape, vals):
        """
        vals -> {regime_0:value_0, regime_1:value_1}

        returns a placeholder, whilst taking care of the feed dict
        required when doing different things. For example whilst training
        with dropout want keep_prob=0.5 and in validation keep_prob=0.5.
        The regime would then be {'training':0.5, 'validation':1.}
        """
        place = tf.placeholder(dtype, shape)
        for regime in vals:
            if regime in self.feeds.keys():
                self.feeds[regime][place] = vals[regime]
            else:
                self.feeds[regime] = {place: vals[regime]}
        return place

    def feed(self, regime):
        """get feed dict with parameters from smart placeholders"""
        try:
            return self.feeds[regime]
        except KeyError:
            return {}


