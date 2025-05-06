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

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# own package
from mypkg.koopman.models.deepkoopman import KoopmanAutoencoder
from mypkg.utils.datapreprocess import tfds2numpy, df2tfds, tfds2tensor
from mypkg.utils.load_model import load_model
from mypkg.train import load_and_train_model
from auv.test import test

## ============================================ Data section ==========================================
# Data split 
raw_data = pd.read_csv("auv/data/processed/stasmc_fault_T3_curve_spiral_adaptive_data_6.csv")
time = raw_data["time"].values

# Data preprocessing
eta = raw_data.loc[:, [f"eta_{i}" for i in range(6)]]
nu = raw_data.loc[:, [f"nu_{i}" for i in range(6)]]
tau = raw_data.loc[:, [f"tau_{i}" for i in range(6)]]
state = pd.concat([eta, nu], axis=1)
dataset = pd.concat([state, tau], axis=1)

# Convert to tf.data.Dataset using in tf.keras.utils.split_dataset
dataset = df2tfds(dataset)
train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.8, right_size=None, shuffle=False, seed=None)

# Plot the data in 3D
# Select eta_0, eta_1, eta_2 as x, y, z
x = state["eta_0"].to_numpy()
y = state["eta_1"].to_numpy()
z = state["eta_2"].to_numpy()

train_plot = tfds2numpy(train_dataset)
test_plot = tfds2numpy(test_dataset)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='3D trajectory (eta_0, eta_1, eta_2)')
ax.plot(train_plot[:,0], train_plot[:,1], train_plot[:,2], label='3D trajectory (x_train, y_train, z_train)', linestyle='--')
ax.plot(test_plot[:,0], test_plot[:,1], test_plot[:,2], label='3D trajectory (x_test, y_test, z_test)', linestyle='--')
ax.set_xlabel('eta_0')
ax.set_ylabel('eta_1')
ax.set_zlabel('eta_2')
ax.legend()

plt.show()

## ============================================ Model section ==========================================
# Define the model parameters
n_state = 12
n_control = 6
encoder_hidden_layers = { 'e1': (16,'relu'), 'e2': (32,'relu'), 'e3': (64,'relu'), 'e4': (128,'relu'), 'e5': (256,'relu') }
decoder_hidden_layers = { 'd1': (256,'relu'), 'd2': (128,'relu'), 'd3': (64,'relu'), 'd4': (32,'relu'), 'd5': (16,'relu') }

# Create the model and load the weights
model = KoopmanAutoencoder(n_state, n_control, encoder_hidden_layers, decoder_hidden_layers)
model.build(None)  # Build the model without input shape    

## ============================================ Training section ==========================================
# Load the trained model weights 
checkpoint_path = 'auv/checkpoints'             # path to save the model
train_dataset = tfds2tensor(train_dataset)      # convert to tensor

# Training settings
train_settings = {
    'max_epoch': 10000,
    'tol': 1e-3,
    'alpha': [1, 1, 0.3, 1e-9, 1e-9, 1e-9],
    'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-4),
}

# !!! comment out the line below if you don't want to train the model
load_and_train_model(model=model, 
                     train_dataset=train_dataset, 
                     checkpoint_path='auv/checkpoints/', 
                     load_chekpoint_path='auv/checkpoints/epoch_9600.weights.h5', 
                     **train_settings)

## ============================================ Testing section ==========================================
# find last weight path 


# Load the trained model weights

weight_path = 'auv/checkpoints/epoch_9600.weights.h5'
model, epoch = load_model(model, path=weight_path)
model.summary()

# Test the model using the train dataset
xk_true = train_dataset[:-1,:n_state]                             # x[k]
uk_true = train_dataset[:-1,n_state:]                             # u[k]
xkp1_true_train = train_dataset[1:,:n_state]                      # x[k+1]
xkp1_pred_train = model((xk_true, uk_true), training=False)       # x[k+1] = model(x[k], u[k])

# Test the model using the test dataset
test_dataset = tfds2numpy(test_dataset)
xk_true = test_dataset[:-1,:n_state]                            # x[k] 
uk_true = test_dataset[:-1,n_state:]                            # u[k]
xkp1_true_test = test_dataset[1:,:n_state]                      # x[k+1]
xkp1_pred_test = model((xk_true, uk_true), training=False)      # x[k+1] = model(x[k], u[k])

## ============================================ Plotting section ==========================================
# Training 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xkp1_true_train[:,0], xkp1_true_train[:,1], xkp1_true_train[:,2], label='3D trajectory (x_true, y_true, z_true)', linestyle='--')
ax.plot(xkp1_pred_train[:,0], xkp1_pred_train[:,1], xkp1_pred_train[:,2], label='3D trajectory (x_true, y_true, z_true)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()

# Testing 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xkp1_true_test[:,0], xkp1_true_test[:,1], xkp1_true_test[:,2], label='3D trajectory (x_true, y_true, z_true)', linestyle='--')
ax.plot(xkp1_pred_test[:,0], xkp1_pred_test[:,1], xkp1_pred_test[:,2], label='3D trajectory (x_pred, y_pred, z_pred)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()

# Plotting all states
n_state = xk_true.shape[1]
time = time[time.shape[0]-xk_true.shape[0]:]  # Adjust time to match the length of xk_true

# Calculate number of rows and columns
n_cols = 2
n_rows = math.ceil(n_state / n_cols)

plt.figure(figsize=(12, 3 * n_rows))
for i in range(n_state):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.plot(time, xkp1_true_test[:, i], label=f'x{i+1}_True', linewidth=2)
    plt.plot(time, xkp1_pred_test[:, i], label=f'x{i+1}_Predict', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel(f"State {i}")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()