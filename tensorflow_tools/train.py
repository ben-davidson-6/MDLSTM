from tensorflow_tools.constants import *
import tensorflow as tf
from numpy import inf as INF

class Trainer:

    def __init__(self, run_name, training_model, dataset_order):
        self.run_name = run_name
        self.training_model = training_model

        self.training_dataset_order = dataset_order
        self.dataset_steps = {name:0 for name in self.training_model.dataset.tf_datasets}
        for x in self.training_dataset_order:
            assert x in self.training_model.dataset.tf_datasets

        self.summary_writers = dict()
        self.create_summary_writers()

        self.steps = {name:0 for name in self.training_model.dataset.tf_datasets}
        self.epoch = 0
        self.best_loss = None
        self.early_stop_dataset = None

    def create_summary_writers(self):
        run_log_dir = os.path.join(LOG_DIR, self.run_name)
        for name in self.training_model.dataset.tf_datasets:
            self.summary_writers[name] = tf.summary.FileWriter(run_log_dir + name)

    def dataset_epoch(self, dataset_name):
        """
        run through all data in interator (until iterator returns OutOfRangeError)
        the operations ran at each iteration are the same for all datasets, except
        those for which dataset.is_training_set(dataset) == True, in this case
        a training step is run as well. Summaries are written by distinct writers,
        one for each dataset, so that graphs etc appear in the same chart
        """
        assert dataset_name in self.training_model.dataset.tf_datasets
        # connect correct inputs
        self.training_model.connect_inputs(dataset_name)
        operation_dict = self.training_model.get_epoch_operations(dataset_name)
        while True:
            try:
                run_output = self.training_model.run_operations(operation_dict, dataset_name)
                self.write_summary(run_output['summaries'], dataset_name)
                self.steps[dataset_name] += 1
            except tf.errors.OutOfRangeError:
                break
        self.end_of_epoch_log_and_clean(dataset_name)

    def write_end_epoch_summaries(self, dataset_name):
        run_output = self.training_model.get_epoch_summs(dataset_name)
        self.write_summary(run_output['summaries'], dataset_name, epoch=True)

    def save_weights(self, dataset_name):
        """
        If early stopping then only save weights if the average loss this
        epoch is better than the previous epoch on the validation set, otherwise
        just keep saving the weights
        """
        if dataset_name == self.early_stop_dataset and self.early_stop_dataset is not None:
            epoch_loss = self.training_model.get_avg_loss()
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss if epoch_loss < self.best_loss else self.best_loss
                self.training_model.save(self.save_path(), global_step=self.epoch)
        elif self.early_stop_dataset is None and dataset_name == self.training_dataset_order[-1]:
            self.training_model.save(self.save_path())

    def end_of_epoch_log_and_clean(self, dataset_name):
        """
        ran at the end of every epoch, write the summaries
        potentially save the weights,
        reset metric values
        """
        self.write_end_epoch_summaries(dataset_name)
        self.save_weights(dataset_name)
        self.training_model.reinitialise_metrics(dataset_name)

    def save_path(self):
        return os.path.join(MODEL_FILE_DIR, self.run_name, 'model')

    def train(self, epochs, restore_path=None, exclude_vars=None):
        """
        train the model for the given number of epochs, potentially starting from
        the restore path weights. This runs epochs in the order given when initialising
        the object
        """
        self.training_model.get_first_weights(
            restore_path=restore_path,
            exclude_vars=exclude_vars)
        while self.epoch < epochs:
            for dataset_name in self.training_dataset_order:
                self.dataset_epoch(dataset_name)
            self.epoch += 1
        self.training_model.sess.close()
        del self.training_model

    def add_early_stopping(self, dataset_name):
        self.best_loss = INF
        assert dataset_name in self.training_dataset_order
        self.early_stop_dataset = dataset_name

    def write_summary(self, summary_output, dataset_name, epoch=False):
        step = self.steps[dataset_name] if not epoch else self.epoch
        for out in summary_output:
            self.summary_writers[dataset_name].add_summary(out, step)
            if epoch:
                self.summary_writers[dataset_name].flush()
