import os


ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_FILE_DIR = os.path.join(DATA_DIR, 'models')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
MNIST_DATASET_PATH = os.path.join(DATASET_DIR, 'mnist.npz')

TRAIN_STEP = 'train_step'
INPUT_TENSOR = 'inputs'
INPUT_TENSOR_0 = 'inputs:0'
LOSS_TENSOR = 'loss'
LOSS_TENSOR_0 = 'loss:0'
LEARNING_RATE_0 = 'learning_rate:0'
LEARNING_RATE = 'learning_rate'
AVG_LOSS = 'average_loss'
AVG_LOSS_0 = 'average_loss:0'