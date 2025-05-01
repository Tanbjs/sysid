import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# own package
from koopman.models.deepkoopman import KoopmanAutoencoder
from koopman.utils.datapreprocess import csv2tfds, tfds2numpy, tfds2tensor
from inverted_pendulum.train import train
from inverted_pendulum.test import test


# Define the model parameters
n_state = 4
n_control = 1
encoder_hidden_layers = { 'e1': (8,'relu'), 'e2': (16,'relu'), 'e3': (32,'relu'), 'e4': (64,'relu'), 'e5': (128,'relu') }
decoder_hidden_layers = { 'd1': (128,'relu'), 'd2': (64,'relu'), 'd3': (32,'relu'), 'd4': (16,'relu'), 'd5': (8,'relu') }

# Create the model and load the weights
model = KoopmanAutoencoder(n_state, n_control, encoder_hidden_layers, decoder_hidden_layers)
# Must build the model before loading weights
model.build(None)
model.load_weights('inverted_pendulum/checkpoints/epoch_1000.weights.h5')

# data split 
raw_data = csv2tfds('inverted_pendulum/data/raw/Closed_loop_IVP_with_white_noise.csv')
train_dataset, test_dataset = tf.keras.utils.split_dataset(raw_data, left_size=0.8, right_size=None, shuffle=False, seed=None)

# Training settings (comment when using trained model to inference)
checkpoint_path = 'inverted_pendulum/checkpoints'
# train(model, train_dataset, checkpoint_path)

# Inference model
test_dataset = tfds2numpy(test_dataset)
x_true = test_dataset[:,:n_state]
u_true = test_dataset[:,n_state:]
x_pred = model((x_true, u_true), training=False)

# plot
n_state = x_true.shape[1]
time = range(x_true.shape[0])   # time index

plt.figure(figsize=(12, 2.5 * n_state))

for i in range(n_state):
    plt.subplot(n_state, 1, i + 1)
    plt.plot(time , x_true[:, i], label=f'x{i+1}_True', linewidth=2)
    plt.plot(time , x_pred[:, i], label=f'x{i+1}_Predict', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel(f"State {i}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()