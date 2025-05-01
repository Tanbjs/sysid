import os
import tensorflow as tf

from koopman.models.deepkoopman import KoopmanAutoencoder
from koopman.utils.loss import compute_loss

def batch_train(model: KoopmanAutoencoder, dataset: list, alpha, optimizer, device):

    batch_size = len(dataset[0])
    for i in range(len(dataset)):
        if len(dataset[i]) != batch_size:
            break
        batch = dataset[i]
        x = tf.convert_to_tensor(batch[:,:,:model.n_state], dtype=tf.float32)
        u = tf.convert_to_tensor(batch[:,:,model.n_state:], dtype=tf.float32)
        with tf.device(device), tf.GradientTape() as tape:
            loss_batch = compute_loss(model=model, x_batch=x, u_batch=u, alpha=alpha)
            loss = tf.reduce_sum(loss_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model, loss

