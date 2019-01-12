import tensorflow as tf


def weighted_dice_binary(gt_tensor, pred_tensor):
    """
        use a balanced loss with

        (1-ratio)*lossTerm(pixel_belongin_to_cell) + (ratio)*lossTerm(pixel_background)

        this will penalise incorrect cells more than incorrect background (if ratio < 0.5)
        and as such we get some class balancing. If we use the original ratio withou
        thresholding the network is too keen to predict cells at any cost as this ratio
        can be 0.

    """
    _, height, width, inp_size = gt_tensor.get_shape().as_list()
    batch_size = tf.shape(gt_tensor)[0]
    # reshape labels so we have 2D also  batch x height * width x 2
    reshaped_labels = tf.reshape(gt_tensor, [-1, height * width, inp_size])

    preds = tf.nn.softmax(logits=pred_tensor)
    reshaped_preds = tf.reshape(preds, [-1, height * width, inp_size])

    multed = tf.reduce_sum(reshaped_labels * reshaped_preds, axis=1)
    summed = tf.reduce_sum(reshaped_labels + reshaped_preds, axis=1)

    r0 = tf.reduce_sum(reshaped_labels[:, :, 0], axis=1)
    r1 = tf.cast(height * width, tf.float32) - r0
    w0 = 1. / (r0 * r0 + 1.)
    w1 = 1. / (r1 * r1 + 1.)

    numerators = w0 * multed[:, 0] + w1 * multed[:, 1]
    denom = w0 * summed[:, 0] + w1 * summed[:, 1]

    dices = 1. - 2. * numerators / denom
    loss = tf.reduce_mean(dices)

    # vars = model.graph.get_collection('trainable_variables')
    # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
    #                    if 'bias' not in v.name]) * 0.001
    return loss  # + lossL2