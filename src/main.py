from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorflow_federated import python as tff

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))
])
MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))
])
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)
SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


@tff.tf_computation(tff.SequenceType(tf.float32))
def get_local_temperature_average(local_temperatures):
    sum_and_count = (local_temperatures.reduce((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1)))
    return sum_and_count[0] / tf.to_float(sum_and_count[1])


@tff.federated_computation(tff.FederatedType(tff.SequenceType(tf.float32), tff.CLIENTS))
def get_global_temperature_average(sensor_readings):
    return tff.federated_mean(tff.federated_map(get_local_temperature_average, sensor_readings))


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255. for i in batch_samples], dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })

    return output_sequence


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    y_hat = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(tf.one_hot(batch.y, 10) * tf.log(y_hat), reduction_indices=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # define a group of model variables and set them to `initial_model`
    model_vars = tff.utils.get_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    # perform one step of gradient descent using loss from `batch_loss`
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # mapping function to apply to each batch
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    # TODO: Replace with `tff.sequence_average` once implemented
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches
        )
    )


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data])
    )


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    """
    The simplest way to implement federated training is to train locally, and then average the models.

    Note that in the full-featured implementation of Federated Averaging provided by tff.learning,
    rather than averaging the models, we prefer to average model deltas, for a number of reasons,
    e.g., the ability to clip the update norms, for compression, etc.
    """
    return tff.federated_mean(
        tff.federated_map(
            local_train,
            [tff.federated_broadcast(model), tff.federated_broadcast(learning_rate), data]
        )
    )


# noinspection PyTypeChecker
def run(constants):
    tf.enable_resource_variables()

    # simulate federated input
    print(get_global_temperature_average([[68., 70.], [71.], [68., 72., 70.]]))

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
    federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

    initial_model = {
        'weights': np.zeros([784, 10], dtype=np.float32),
        'bias': np.zeros([10], dtype=np.float32)
    }
    sample_batch = federated_train_data[5][-1]
    print(batch_loss(initial_model, sample_batch))

    model = initial_model
    losses = []
    for _ in range(5):
        model = batch_train(model, sample_batch, 0.1)
        losses.append(batch_loss(model, sample_batch))

    print(losses)

    locally_trained_model = local_train(initial_model, 0.1, federated_train_data[5])

    print('initial_model loss =', local_eval(initial_model, federated_train_data[5]))
    print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[5]))

    print()
    print('initial_model loss =', local_eval(initial_model, federated_train_data[0]))
    print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[0]))

    print()
    print('initial_model loss =', federated_eval(initial_model, federated_train_data))
    print('locally_trained_model loss =', federated_eval(locally_trained_model, federated_train_data))

    # Federated Training
    print()
    print('Federated Training')
    model = initial_model
    learning_rate = 0.1
    for epoch in range(5):
        model = federated_train(model, learning_rate, federated_train_data)
        learning_rate = learning_rate * 0.9
        loss = federated_eval(model, federated_train_data)
        print('epoch {}, loss {}'.format(epoch, loss))

    print()
    print('initial_model test loss =', federated_eval(initial_model, federated_test_data))
    print('trained_model test loss =', federated_eval(model, federated_test_data))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Federated Learning Example')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
