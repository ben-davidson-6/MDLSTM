from tensorflow_tools.constants import *
from tensorflow_tools.inference import ForwardModel, InferenceBuilder
from tensorflow_tools.training_model import TrainingModel
from tensorflow_tools.train import Trainer
from layers.loss_fn import weighted_dice_binary
from networks.mdlstm import CircleData, MDLSTMNet

import matplotlib.pyplot as plt
import tensorflow as tf
import sys


def train_model():

    net = MDLSTMNet(units=16)
    dataset = CircleData()
    model = TrainingModel(net, dataset)

    model.build_network()

    def loss_fn():
        g = tf.get_default_graph()
        gt_tensor = dataset.input_tensors['labels']
        pred = g.get_tensor_by_name('out_softmax:0')
        loss = weighted_dice_binary(gt_tensor, pred)
        return loss

    model.add_loss(loss_fn=loss_fn)

    def lr_schedule(init_learn, global_step):
        return tf.train.cosine_decay_restarts(
            init_learn,
            global_step,
            4000,
            m_mul=0.5)

    def optimiser(learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate)

    optimiser = model.build_optimisation(
        optimiser,
        init_learn=0.1,
        learn_scheduler=lr_schedule)

    model.add_optimiser(optimiser)

    def output_summary():
        g = tf.get_default_graph()
        prediction = g.get_tensor_by_name('prediction:0')
        prediction = tf.expand_dims(prediction, 3)
        prediction = tf.cast(prediction, tf.float32)
        labels = dataset.input_tensors['labels'][:, :, :, 1:]
        image = dataset.input_tensors['image']
        summary = tf.concat([image, prediction, labels], axis=2)

        return tf.summary.image('Network Learning', summary, max_outputs=1)

    model.add_standard_loss_summaries()
    model.add_intra_epoch_summary(output_summary)

    run_name = 'mdlstm'
    trainer = Trainer(run_name, model, ['validation', 'training'])
    trainer.add_early_stopping(dataset_name='validation')
    trainer.train(epochs=5)


def show_circle_model():
    model_path = os.path.join(MODEL_FILE_DIR, 'mdlstm', 'optimised_model.pb')
    outputs = {'pred':'prediction'}
    f = ForwardModel(model_path, outputs)

    for im, gt in CircleData._generator(10):
        net_out = f({'image':im})
        pred = net_out['pred'][0]

        plt.subplot(2, 1, 1)
        plt.imshow(pred)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(gt[:, :, 1])
        plt.axis('off')
        plt.show()


def append_dims(inputs):
    inputs['image'] = tf.expand_dims(inputs['image'], 0)
    return inputs


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train_model()
    elif sys.argv[1] == 'inspect':
        # build only forward model
        run_name = 'mdlstm'
        model_step = 4
        net = MDLSTMNet(units=16)
        builder = InferenceBuilder(run_name, model_step, net)

        inputs = {'image': {'dtype': tf.float32, 'shape': [256, 256, 1]}}
        output_node_name = 'prediction'
        builder.to_optmised_graph(output_node_name, inputs,  preprocess=append_dims, keep_prob=1.)

        # show in use
        show_circle_model()

