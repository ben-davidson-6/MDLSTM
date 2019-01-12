import tensorflow as tf
from tensorflow import TensorArray

# number cells to run top left to bot righ top right to bot left ...
directions = 4

@tf.custom_gradient
def clip_grad_layer(x):
    def grad(dy):
        return tf.clip_by_global_norm([dy], 5.)[0][0]
    return tf.identity(x), grad


def build_diagonal_cell_weights(units, input_channels, initialisation_sigma):
    # filters for depthwise convolution
    # [filter_height, filter_width, in_channels, channel_multiplier]
    number_diagonal_values_to_convolve = 2
    hidden_weights = tf.get_variable(
        'hidden_weights',
        initializer=tf.random_normal(
            [number_diagonal_values_to_convolve, units, directions, 5 * units],
            0,
            initialisation_sigma))

    number_diagonal_values_to_convolve = 1
    input_weights = tf.get_variable(
        'input_weights',
        initializer=tf.random_normal(
            [number_diagonal_values_to_convolve, input_channels, directions, 5 * units],
            0,
            initialisation_sigma))

    def _bias_weights(name, forget=False):
        if forget:
            return tf.get_variable(name, initializer=tf.ones([units, directions]))
        else:
            return tf.get_variable(name, initializer=tf.zeros([units, directions]))

    ib = _bias_weights('i_bias')
    fb1 = _bias_weights('f1_bias')
    fb2 = _bias_weights('f2_bias')
    cb = _bias_weights('c_bias')
    ob = _bias_weights('o_bias')
    return input_weights, hidden_weights, (ib, fb1, fb2, cb, ob)


def diagonal_lstm(units, input_channels, cell_computation, initialisation_sigma):
    input_weights, hidden_weights, (ib, fb1, fb2, cb, ob) = build_diagonal_cell_weights(
        units, input_channels, initialisation_sigma)

    def cell(diagonal_input, diagonal_acti, diagonal_cell):

        """
        Input
            diagonal_input
                b x diag_len x in_size x 4
            diagonal_activation:
                b x diag_len_prev + 1 x units x 4
            diagonal_cell:
                b x diag_len_prev + 1 x units x 4
        Return
            tensor of activation
                b x d x units x 4
        """

        _, diagonal_size, _, _ = diagonal_input.get_shape().as_list()
        # [batch,prev_diag+1,units,4*units] -> [batch,prev_diag,1,dirs*(5units)]
        hidden_mats = tf.nn.depthwise_conv2d(diagonal_acti, hidden_weights, [1, 1, 1, 1], 'VALID')
        hidden_mats = hidden_mats[:, :, 0, :]
        # [batch,diag,in_size,4*input_channels] -> [batc,diag,1,dirs*(5units)]
        input_mats = tf.nn.depthwise_conv2d(diagonal_input, input_weights, [1, 1, 1, 1], 'VALID')
        input_mats = input_mats[:, :, 0, :]

        # combined mats
        # [b,diag,dirs(5units)]
        # [b,diagonal,5units,dirs]
        # [b,diagonal,units,dirs,5]
        combined = input_mats + hidden_mats
        combined = tf.stack(tf.split(combined, axis=2, num_or_size_splits=directions), 3)
        combined = tf.stack(tf.split(combined, axis=2, num_or_size_splits=5), 4)

        # getting relavent cell states at each pixel
        cell_up = diagonal_cell[:, 0:-1, :, :]
        cell_left = diagonal_cell[:, 1:, :, :]

        i = tf.sigmoid(combined[:, :, :, :, 0] + ib)
        f1 = tf.sigmoid(combined[:, :, :, :, 1] + fb1)
        f2 = tf.sigmoid(combined[:, :, :, :, 2] + fb2)
        o = tf.sigmoid(combined[:, :, :, :, 3] + ob)
        cell = combined[:, :, :, :, 4] + cb

        if cell_computation == 'leaky':
            cell_state = tf.tanh(cell) * i + ((cell_up * f1 + cell_left * f2) / (f1 + f2)) * (1 - i)
        elif cell_computation == 'graves':
            cell_state = tf.tanh(cell) * i + (cell_up * f1 + cell_left * f2)
        else:
            raise ValueError('No cell named {}'.format(cell_computation))

        # [b,diagonal,unit,dir]
        activation = clip_grad_layer(o * tf.tanh(cell_state))
        cell_state = clip_grad_layer(cell_state)
        return activation, cell_state

    return cell


def get_linear_diagonal_indices(height, width, diagonal):
    """
    In 2d image get linear indices of the
    reverse diagonal eg diagonal 1
    o * o
    * o o
    o o o
    """
    width_offset = 1
    height_offset = width
    delta = -height_offset + width_offset
    start = tf.cond(
        diagonal < height,
        lambda: diagonal*height_offset,
        lambda: (height - 1)*height_offset + (diagonal - height + 1)*width_offset)
    end = tf.cond(
        diagonal < width,
        lambda: diagonal*width_offset - 1,
        lambda: (width - 1) + (diagonal - (width - 1))*height_offset - 1)
    inds = tf.cond(
        tf.equal(start, end),
        lambda: tf.expand_dims(start, axis=0),
        lambda: tf.range(start, end, delta))
    inds = tf.cast(inds, tf.int32)
    return inds


def get_multi_diagonal_indices(height, width, diagonal):
    """
    Instead of linear indices, gives (row, col) of
    reverse diagonal
    [ row, col
      row, col...]
    """
    linear_inds = get_linear_diagonal_indices(height, width, diagonal)
    rows = tf.math.floor(linear_inds / width)
    rows = tf.cast(rows, tf.int32)
    cols = linear_inds - rows*width
    mult_inds = tf.stack([rows, cols], axis=1)
    return mult_inds


def get_diagonal_values_from_multi(tensor, indices):
    """
    uses multi index values [row, col]
    to get the actual values in the diagonal
    of tensor

    gather_nd works using first indices you give it
    so we assume tensor is of shape
        [height, width, batch, inp_size, direction]
    """
    values = tf.gather_nd(tensor, indices)
    # reshapes to [batch, diagonal, inp_szie, direction]
    return tf.transpose(values, (1, 0, 2, 3))


def get_initial_values(batch, height, width, units,):
    num_diag = height + width - 1
    zeros = tf.stack([batch, 2, units, directions])
    current_activations = tf.fill(zeros, 0.0)
    initial_state = tf.fill([batch, 1, units, directions], 0.0)
    current_states = tf.tile(initial_state, [1, 2, 1, 1])
    diagonal = tf.constant(0)
    return num_diag, initial_state, current_activations, current_states, diagonal


def initialise_tensor_arrays(height, width, num_diag, units):
    linear_inds_ta = TensorArray(
        dtype=tf.int32,
        size=num_diag,
        element_shape=tf.TensorShape([None]),
        clear_after_read=False,
        name='linear_inds',
        infer_shape=False)

    multi_inds_ta = TensorArray(
        dtype=tf.int32,
        size=num_diag,
        element_shape=tf.TensorShape([None, 2]),
        clear_after_read=False,
        name='mult_inds',
        infer_shape=False)

    activations_ta = TensorArray(
        dtype=tf.float32,
        size=height * width,
        element_shape=tf.TensorShape([None, units, 4]))

    return linear_inds_ta, multi_inds_ta, activations_ta


def write_diagonal_inds_to_tensor_array(height, width, num_diag, linear_inds_ta, multi_inds_ta):
    for d in range(num_diag):
        linear_tensor = get_linear_diagonal_indices(height, width, tf.constant(d))
        multi_tensor = get_multi_diagonal_indices(height, width, tf.constant(d))
        linear_inds_ta = linear_inds_ta.write(d, linear_tensor)
        multi_inds_ta = multi_inds_ta.write(d, multi_tensor)
    return linear_inds_ta, multi_inds_ta


def fast_MD_dynamic(input_data, units, cell_computation, initialisation_sigma):
    """
    carries out iteration over diagonals
    input_data = (b,h,w,i,d)
        where d are the 4 direcitons
    units
        number of units in cell
    """
    _, height, width, inp_channels, directions = input_data.get_shape().as_list()
    batch_size = tf.shape(input_data)[0]

    num_diag, initial_state, current_activations, current_states, diagonal = get_initial_values(
        batch_size, height, width, units
    )
    input_data_transposed = tf.transpose(input_data, (1, 2, 0, 3, 4))
    cell = diagonal_lstm(units, inp_channels, cell_computation, initialisation_sigma)

    linear_inds_ta, multi_inds_ta, activations_ta = initialise_tensor_arrays(
        height, width, num_diag, units)

    linear_inds_ta, multi_inds_ta = write_diagonal_inds_to_tensor_array(
        height, width, num_diag, linear_inds_ta, multi_inds_ta)

    def pad_diagonal(x, h):
        d_x = tf.shape(x)[1]
        d_h = tf.shape(h)[1]
        to_pad = d_x - d_h + 1

        padded_h = tf.cond(
            tf.equal(to_pad, 0),
            lambda: h,
            lambda: tf.cond(
                tf.equal(to_pad, 1),
                lambda: tf.pad(h, [[0, 0], [0, 1], [0, 0], [0, 0]]),
                lambda: tf.pad(h, [[0, 0], [1, 1], [0, 0], [0, 0]]),
            )
        )
        return padded_h

    def body(activations_ta, current_activations, current_states, diagonal):
        # Get the diagonal values of the input
        # [b,d,inp_channel,direction]
        input_diagonal = get_diagonal_values_from_multi(
            input_data_transposed,
            multi_inds_ta.read(diagonal))

        # pad so have required number of previous states
        current_states = pad_diagonal(input_diagonal, current_states)
        current_activations = pad_diagonal(input_diagonal, current_activations)

        # [batch,diagonal,unit,direction]
        current_activations, current_states = cell(input_diagonal, current_activations, current_states)
        current_states.set_shape([None, None, units, directions])
        current_activations.set_shape([None, None, units, directions])

        # [batch,units,direction]
        activations_ta = activations_ta.scatter(
            linear_inds_ta.read(diagonal),
            tf.transpose(current_activations, (1, 0, 2, 3)))

        diagonal += 1
        return activations_ta, current_activations, current_states, diagonal

    def cond(activations_ta, current_activations, current_states, diagonal):
        return diagonal < num_diag

    acti_shape = tf.TensorShape([None, None, units, directions])
    cell_shape = tf.TensorShape([None, None, units, directions])
    diag_shape = tf.TensorShape([])
    ta_shape = tf.TensorShape(None)
    returned = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[activations_ta, current_activations, current_states, diagonal],
        name='loop',
        shape_invariants=[ta_shape, acti_shape, cell_shape, diag_shape],
        swap_memory=True)

    activations = returned[0].stack()
    activations.set_shape([height * width, None, units, directions])
    activations = tf.transpose(activations, (1, 0, 2, 3))
    activations = tf.split(activations, num_or_size_splits=height, axis=1)
    activations = tf.stack(activations, 1)

    return activations


def MD_parallel(image, units, cell_computation, initialisation_sigma):
    _, height, width, inp_size = image.get_shape().as_list()

    # four orientations
    tl = image
    tr = tf.map_fn(tf.image.flip_left_right, image)
    bl = tf.map_fn(tf.image.flip_up_down, image)
    br = tf.map_fn(tf.image.flip_left_right, tf.map_fn(tf.image.flip_up_down, image))
    all_together = tf.stack([tl, tr, bl, br], 4)

    # [b,height,width,units,dir]
    all_activations = fast_MD_dynamic(all_together, units, cell_computation, initialisation_sigma)
    tl, tr, bl, br = tf.split(all_activations, num_or_size_splits=4, axis=4)
    tl = tl[:, :, :, :, 0]
    tr = tf.map_fn(tf.image.flip_left_right, tr[:, :, :, :, 0])
    bl = tf.map_fn(tf.image.flip_up_down, bl[:, :, :, :, 0])
    br = tf.map_fn(tf.image.flip_up_down, tf.map_fn(tf.image.flip_left_right, br[:, :, :, :, 0]))
    all_together = tf.stack([tl, tr, bl, br], 4)
    all_together.set_shape([None, height, width, units, 4])

    return all_together
