import tensorflow as tf
from tensorflow_tools.network import Network
from tensorflow_tools.dataset import Dataset
from tensorflow_tools.constants import *


class TrainingModel:
    def __init__(self, network, dataset):
        assert isinstance(network, Network)
        assert isinstance(dataset, Dataset)
        self.network = network
        self.dataset = dataset
        self.graph = tf.Graph()
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True),
            graph=self.graph)
        self.saver = None

        self.losses = []
        self.intra_summaries = []
        self.epoch_summaries = []
        self.metric_updates = []
        self.metrics = []
        self.reset_ops = []

    def _build_input_pipeline(self):
        with tf.device('/cpu:0'):
            self.dataset.build_pipeline()
        return self.dataset.get_inputs()

    def build_network(self,):
        with self.graph.as_default():
            inputs = self._build_input_pipeline()
            self.network.build_network(inputs)

    def build_on_gpu(self, split_inputs, gpu):
        # assumes all variables are built on cpu
        # in build_network
        with tf.device('/gpu:{}'.format(gpu)):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                tower_out = self.network.build_network(split_inputs[gpu])
        return tower_out

    def build_network_multiple(self, gpus, name):
        tower_out = []
        with self.graph.as_default():
            inputs = self._build_input_pipeline()
            with tf.device('/cpu:0'):
                split_inputs = self.dataset.gpu_batch(inputs, gpus)
            for gpu in range(gpus):
                tower_out.append(self.build_on_gpu(split_inputs, gpu))
            with tf.device('/cpu:0'):
                tf.concat(tower_out, axis=0, name=name)

    def add_loss(self, loss_fn):
        with self.graph.as_default():
            loss = loss_fn()
            tf.identity(loss, name=LOSS_TENSOR)
        self.losses.append(loss.name)

    def get_loss(self,):
        return self.graph.get_tensor_by_name(LOSS_TENSOR_0)

    def learning_rate_schedule(self, init_lr, learn_scheduler):

        with self.graph.as_default():
            global_step = tf.train.get_or_create_global_step()
            learning_rate = learn_scheduler(init_lr, global_step)
        return learning_rate

    def _build_learning_rate(self, init_learn, learn_scheduler):
        if learn_scheduler is not None:
            learning_rate = self.learning_rate_schedule(init_learn, learn_scheduler)
        else:
            learning_rate = init_learn
        learning_rate = tf.identity(learning_rate, name=LEARNING_RATE)
        return learning_rate

    def build_optimisation(self, optimiser, init_learn, learn_scheduler=None):
        learning_rate = self._build_learning_rate(init_learn, learn_scheduler)
        with self.graph.as_default():
            optimiser = optimiser(learning_rate)
        return optimiser

    def _vars_to_minimise(self, vars_to_minimise):
        if vars_to_minimise is not None:
            trainable_vars = self.graph.get_collection('trainable_variables')
            vars_to_minimise = [x for x in trainable_vars if any([y in x.name for y in vars_to_minimise])]
        else:
            vars_to_minimise = None
        return vars_to_minimise

    def add_optimiser(self, optimiser, vars_to_minimise=None, multi_gpu=False):
        loss = self.get_loss()
        with self.graph.as_default():
            vars_to_minimise = self._vars_to_minimise(vars_to_minimise)
            global_step = tf.train.get_or_create_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimiser.minimize(
                    loss,
                    name=TRAIN_STEP,
                    var_list=vars_to_minimise,
                    global_step=global_step,
                    colocate_gradients_with_ops=multi_gpu)

    def add_intra_epoch_summary(self, summary_fnc):
        with self.graph.as_default():
            tensor = summary_fnc()
            self.intra_summaries.append(tensor.name)

    def add_epoch_summaries(self, summary_fnc):
        with self.graph.as_default():
            tensor = summary_fnc()
            self.epoch_summaries.append(tensor.name)

    def _metric_name_to_scope(self, metric_name):
        return metric_name + '_scope'

    def _get_metric_value(self, metric_name):
        metric_scope = self._metric_name_to_scope(metric_name)
        total = self.graph.get_tensor_by_name(metric_scope + '/' + 'total:0')
        count = self.graph.get_tensor_by_name(metric_scope + '/' + 'count:0')
        return total / count

    def _build_metric(self, metric_name, metric_func):
        metric_scope = self._metric_name_to_scope(metric_name)

        ma, update = metric_func(metric_scope)
        reset_metric = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES,
            scope=metric_scope)
        reset = tf.variables_initializer(var_list=reset_metric)

        to_track = self._get_metric_value(metric_name)
        to_track = tf.identity(to_track, name=metric_name)
        self.metrics.append(to_track.name)

        return reset, update

    def add_metric(self, metric_func, metric_name):
        with self.graph.as_default():
            reset, update = self._build_metric(metric_name, metric_func)
            self.metric_updates.append(update.name)
            self.reset_ops.append(reset.name)

        def metric_summary():
            to_watch = self.graph.get_tensor_by_name(metric_name + ':0')
            return tf.summary.scalar(metric_name, to_watch)

        self.add_epoch_summaries(metric_summary)

    def add_standard_loss_summaries(self):
        def loss_intra_summary():
            to_watch = self.graph.get_tensor_by_name(LOSS_TENSOR_0)
            return tf.summary.scalar('loss_intra_epoch', to_watch)

        def loss_epoch_summary():
            to_watch = self.graph.get_tensor_by_name(AVG_LOSS_0)
            return tf.summary.scalar('avg_loss_epoch', to_watch)

        def learn_rate_summary():
            to_watch = self.graph.get_tensor_by_name(LEARNING_RATE_0)
            return tf.summary.scalar('learning_rate', to_watch)

        def metric_func(name):
            loss = self.graph.get_tensor_by_name(LOSS_TENSOR_0)
            return tf.metrics.mean(loss, name=name)

        self.add_intra_epoch_summary(loss_intra_summary)
        self.add_metric(metric_func, AVG_LOSS)
        self.add_intra_epoch_summary(learn_rate_summary)

    def get_metric_values(self):
        return [self.graph.get_operation_by_name(name) for name in self.metrics]

    def get_metric_reset_ops(self):
        return [self.graph.get_operation_by_name(name) for name in self.reset_ops]

    def get_intra_summaries(self):
        return [self.graph.get_tensor_by_name(name) for name in self.intra_summaries]

    def get_epoch_summaries(self):
        return {'summaries': [self.graph.get_tensor_by_name(name) for name in self.epoch_summaries]}

    def get_metric_updates(self):
        return [self.graph.get_tensor_by_name(name) for name in self.metric_updates]

    def get_train_step(self,):
        return self.graph.get_operation_by_name(TRAIN_STEP)

    def get_epoch_operations(self, dataset_name):
        # if training epoch get train step
        training_epoch = self.dataset.is_training_set(dataset_name)
        train_step = [self.get_train_step()] if training_epoch else []
        fetches = {
            'train_step': train_step,
            'summaries': self.get_intra_summaries(),
            'metric_updates': self.get_metric_updates()}
        return fetches

    def _get_vars_in_restore_path(self, restore_path, exclude_vars):
        to_restore = [x[0] for x in tf.train.list_variables(restore_path)]
        with self.graph.as_default():
            vars_to_restore = [
                x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                if x.name[:-2] in to_restore]
        if exclude_vars is not None:
            vars_to_restore = [
                x for x in vars_to_restore
                if not any([y in x.name[:-2] for y in exclude_vars])]
        return vars_to_restore

    def restore_variables(self, restore_path, exclude_vars):
        # needed in case there are any variables not in restore path
        self.initialise_variables()
        vars_to_restore = self._get_vars_in_restore_path(restore_path, exclude_vars)
        with self.graph.as_default():
            saver = tf.train.Saver(var_list=vars_to_restore)
        saver.restore(self.sess, restore_path)

    def initialise_variables(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        operation_dict = self.get_metric_reset_ops()
        self.sess.run(
            operation_dict)

    def get_first_weights(self, restore_path, exclude_vars):
        if restore_path is None:
            self.initialise_variables()
        else:
            self.restore_variables(restore_path, exclude_vars)
        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def save(self, save_path, global_step):
        self.saver.save(self.sess, save_path, global_step=global_step)

    def get_avg_loss(self):
        avg_loss = self.graph.get_tensor_by_name(AVG_LOSS_0)
        return self.sess.run(avg_loss)

    def connect_inputs(self, dataset_name):
        init = self.graph.get_operation_by_name(dataset_name)
        self.sess.run(init)

    def reinitialise_metrics(self, dataset_name):
        operation_dict = self.get_metric_reset_ops()
        self.sess.run(
            operation_dict,
            feed_dict=self.network.feed(dataset_name))

    def get_epoch_summs(self, dataset_name):
        operation_dict = self.get_epoch_summaries()
        run_output = self.sess.run(
            operation_dict,
            feed_dict=self.network.feed(dataset_name))
        return run_output

    def run_operations(self, operation_dict, dataset_name):
        feed = self.network.feed(dataset_name)
        return self._run_operations(operation_dict, feed)

    def _run_operations(self, operation_dict, feed):
        run_output = self.sess.run(
            operation_dict,
            feed_dict=feed)
        return run_output