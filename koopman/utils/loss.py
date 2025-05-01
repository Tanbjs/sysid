import numpy as np
import tensorflow as tf

def compute_loss(model, alpha, x_batch, u_batch):
    """
    Compute the total loss for the DeepKoopman model.

    Args:
        model: The DeepKoopman model containing encoder, decoder, and Koopman operators.
        alpha: A list of weights for different loss components.
        x_batch: Input state batch.
        u_batch: Input control batch.

    Returns:
        Total loss as a TensorFlow tensor.
    """
    # Calculating each loss
    reconstruction_error_val = reconstruction_error(x_batch, u_batch, model)
    multi_step_pred_error_val = multi_step_pred_error(x_batch, u_batch, model)
    k_linear_error_val = linear_error(x_batch, u_batch, model)

    L_ox = tf.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(reconstruction_error_val, axis=-1)), axis=-1)
    L_xx = tf.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(multi_step_pred_error_val, axis=-1)), axis=-1)
    L_xo = tf.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(k_linear_error_val, axis=-1)), axis=-1)
    L_inf = (tf.reduce_mean(tf.square(tf.norm(reconstruction_error_val, ord=np.inf, axis=-1)), axis=-1) 
             + tf.reduce_mean(tf.square(tf.norm(multi_step_pred_error_val, ord=np.inf, axis=-1)), axis=-1))

    # L2 regularization on encoder and decoder weights
    reg_enc = tf.add_n([tf.reduce_sum(tf.square(var)) for var in model.encoder.trainable_variables])
    reg_dec = tf.add_n([tf.reduce_sum(tf.square(var)) for var in model.decoder.trainable_variables])

    loss = (alpha[0] * L_ox + alpha[1] * L_xx + alpha[2] * L_xo + alpha[3] * L_inf + alpha[4] * reg_enc + alpha[5] * reg_dec)

    return loss

def multi_step_pred(x_batch, u_batch, model) -> tf.Tensor:
    """
    Perform multi-step prediction using the Koopman operator.

    Args:
        x_batch: Input state batch.
        u_batch: Input control batch.
        model: The DeepKoopman model.

    Returns:
        Multi-step predictions as a TensorFlow tensor.
    """
    p = u_batch.shape[1] - 1
    x_0 = x_batch[:, 0, :]
    enc_0 = model.encoder(x_0)
    # Initial z0
    z = tf.concat([x_0, enc_0], axis=-1)
    z_stack = []
    for i in range(p):
        z = model.A(z) + model.B(u_batch[:, i, :])
        z_stack.append(z)

    return tf.stack(z_stack, axis=1)

def multi_step_pred_error(x_batch, u_batch, model) -> tf.Tensor:
    """
    Compute the multi-step prediction error.

    Args:
        x_batch: Input state batch.
        u_batch: Input control batch.
        model: The DeepKoopman model.

    Returns:
        Multi-step prediction error as a TensorFlow tensor.
    """
    gkpi_pred = multi_step_pred(x_batch, u_batch, model)
    x_p_stack = []
    for i in range(gkpi_pred.shape[1]):
        x_p_stack.append(model.decoder(gkpi_pred[:, i, :]))
    x_p_steps = tf.stack(x_p_stack, axis=1)
    error = x_batch[:, 1:, :] - x_p_steps

    return error

def linear_error(x_batch, u_batch, model) -> tf.Tensor:
    """
    Compute the linearity error of the Koopman operator.

    Args:
        x_batch: Input state batch.
        u_batch: Input control batch.
        model: The DeepKoopman model.

    Returns:
        Linearity error as a TensorFlow tensor.
    """
    gkpi_pred = multi_step_pred(x_batch, u_batch, model)
    linear_stack = []
    for i in range(gkpi_pred.shape[1]):
        enc_i = model.encoder(x_batch[:, i+1, :])
        linear_stack.append(tf.concat([x_batch[:, i+1, :], enc_i], axis=-1))
    gk = tf.stack(linear_stack, axis=1)
    error = gk - gkpi_pred
    return error

def reconstruction_error(x_batch, u_batch, model) -> tf.Tensor:
    """
    Compute the reconstruction error of the model.

    Args:
        x_batch: Input state batch.
        u_batch: Input control batch.
        model: The DeepKoopman model.

    Returns:
        Reconstruction error as a TensorFlow tensor.
    """
    stack = []
    for i in range(x_batch.shape[1]):
        x_t = x_batch[:, i, :]
        enc_t = model.encoder(x_t)
        z_t = tf.concat([x_t, enc_t], axis=-1)
        stack.append(model.decoder(z_t))
    x_hat = tf.stack(stack, axis=1)
    error = x_batch - x_hat

    return error
