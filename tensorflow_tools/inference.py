import tensorflow as tf
from tensorflow_tools.constants import *
from tensorflow_tools.utils import get_all_inputs
# you actually need this for inference as contrib
# is loaded lazily or something
tf.contrib.resampler


class InferenceBuilder:
    def __init__(self, model_name, run, network):
        model_dir= os.path.join(MODEL_FILE_DIR, model_name)
        self.ckpt_path = os.path.join(model_dir, 'model-{}'.format(run))
        self.optimized_pb_path = os.path.join(model_dir, 'optimised_model.pb')
        self.network = network
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def restore_graph(self, inputs, preprocess, **kwargs):
        placeholder_inputs = {}
        with self.graph.as_default():
            for name in inputs:
                dtype = inputs[name]['dtype']
                shape = inputs[name]['shape']
                placeholder_inputs[name] = tf.placeholder(dtype, shape, name=name)

            if preprocess is not None:
                placeholder_inputs = preprocess(placeholder_inputs)

            self.network.build_network(placeholder_inputs, **kwargs)
            saver = tf.train.Saver()
            saver.restore(self.sess, self.ckpt_path)

    def to_optmised_graph(self, outputs, inputs, preprocess=None, **kwargs):
        if type(outputs) != list:
            outputs = [outputs]
        outputs = [x[:-2] if (':0' == x[-2:]) else x for x in outputs]
        self.restore_graph(inputs, preprocess, **kwargs)

        # todo possible white/blacklist uses
        # todo if some variables shouldnt be constant
        print('Warning: making ALL variables in graph constant')
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            self.graph.as_graph_def(),
            outputs
            )
        with tf.gfile.GFile(self.optimized_pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        tf.reset_default_graph()
        self.sess.close()
        del self.graph


class ForwardModel:
    def __init__(self, model_path, outputs):
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.load_graph()

        self.outputs = outputs
        self.build_outputs()

    def load_graph(self):
        with tf.gfile.GFile(self.model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def)

    def build_outputs(self):
        outputs = {}
        for name in self.outputs:
            tensor_name = self.outputs[name]
            tensor_name = tensor_name + ':0' if ':0' != tensor_name[-2:] else tensor_name
            tensor_name = 'import/' + tensor_name
            tensor = self.graph.get_tensor_by_name(tensor_name)
            outputs[name] = tensor
        self.outputs = outputs

    def __call__(self, network_inputs):
        feed_dict = self._build_feed(network_inputs)
        return self.sess.run(self.outputs, feed_dict=feed_dict)

    def _build_feed(self, network_inputs):
        feed_dict = {}
        for name in network_inputs:
            p = self.graph.get_tensor_by_name('import/' + name + ':0')
            feed_dict[p] = network_inputs[name]
        return feed_dict

    def inspect_tensors(self, network_inputs, tensor_names):
        feed_dict = self._build_feed(network_inputs)
        output_tensors = {x: self.graph.get_tensor_by_name('import/' + x) for x in tensor_names}
        return self.sess.run(output_tensors, feed_dict=feed_dict)