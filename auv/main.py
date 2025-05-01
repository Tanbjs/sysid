import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0': all logs, '1': filter INFO, '2': filter WARNING, '3': filter ERROR

import tensorflow as tf
# Check if GPU is available and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU found")
device_name = gpus[0].name
device = device_name[device_name.find("/GPU:0"):]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# own package
from koopman.models.deepkoopman import KoopmanAutoencoder
from koopman.utils.datapreprocess import csv2tfds, tfds2numpy, df2tfds, tfds2tensor
from koopman.utils.load_model import load_model
from auv.train import train
from auv.test import test

# Define the model parameters
n_state = 12
n_control = 6
encoder_hidden_layers = { 'e1': (16,'relu'), 'e2': (32,'relu'), 'e3': (64,'relu'), 'e4': (128,'relu'), 'e5': (256,'relu') }
decoder_hidden_layers = { 'd1': (256,'relu'), 'd2': (128,'relu'), 'd3': (64,'relu'), 'd4': (32,'relu'), 'd5': (16,'relu') }
load_model_status = False

# Create the model and load the weights
model = KoopmanAutoencoder(n_state, n_control, encoder_hidden_layers, decoder_hidden_layers)
model.build(None)  # Build the model without input shape    

# Load the pretrained model weights
model, epoch, load_model_status = load_model(model, 'auv/checkpoints/epoch_800.weights.h5')

# Data split 
raw_data = pd.read_csv("auv/data/processed/stasmc_fault_T3_curve_spiral_adaptive_data_6.csv")
time = raw_data["time"].values
eta = raw_data.loc[:, [f"eta_{i}" for i in range(6)]]
nu = raw_data.loc[:, [f"nu_{i}" for i in range(6)]]
tau = raw_data.loc[:, [f"tau_{i}" for i in range(6)]]
state = pd.concat([eta, nu], axis=1)
dataset = pd.concat([state, tau], axis=1)

# Convert to tf.data.Dataset using in tf.keras.utils.split_dataset
dataset = df2tfds(dataset)
train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.8, right_size=None, shuffle=False, seed=None)

# Training settings
checkpoint_path = 'auv/checkpoints'             # path to save the model
train_dataset = tfds2tensor(train_dataset)      # convert to tensor

max_epoch = 1000
tol = 1e-3
alpha = [1, 1, 0.3, 1e-9, 1e-9, 1e-9]
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# !!!! Comment out the training part if you want to use the pretrained model !!!!

# if load_model_status:
#     train(model=model, dataset=train_dataset, 
#           checkpoint_path=checkpoint_path, epoch=epoch, 
#           max_epoch=max_epoch, tol=tol, alpha=alpha, 
#           optimizer=optimizer, device="/GPU:0")
# else:
#     train(model=model, dataset=train_dataset, 
#         checkpoint_path=checkpoint_path, epoch=1, 
#         max_epoch=max_epoch, tol=tol, alpha=alpha, 
#         optimizer=optimizer, device="/GPU:0")
    
# Inference model
test_dataset = tfds2numpy(test_dataset)
x_true = test_dataset[:,:n_state]
u_true = test_dataset[:,n_state:]
x_pred = model((x_true, u_true), training=False)

# plot
n_state = x_true.shape[1]
time = range(x_true.shape[0])

# Calculate number of rows and columns
n_cols = 2
n_rows = math.ceil(n_state / n_cols)

plt.figure(figsize=(12, 3 * n_rows))

for i in range(n_state):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.plot(time, x_true[:, i], label=f'x{i+1}_True', linewidth=2)
    plt.plot(time, x_pred[:, i], label=f'x{i+1}_Predict', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel(f"State {i}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()