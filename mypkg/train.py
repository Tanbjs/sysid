# system module
import time 
import os
# Path to the main directory
import tensorflow as tf
# own package
from mypkg.koopman.models.deepkoopman import KoopmanAutoencoder
from mypkg.utils.batch_train import batch_train
from mypkg.utils.datapreprocess import generate_train_batch
from mypkg.utils.load_model import load_model

def load_and_train_model(model: KoopmanAutoencoder, train_dataset: tf.Tensor, checkpoint_path: str, load_chekpoint_path: str = None,**train_settings):
    """
    Load the model weights and train the model.
    
    Args:
        model (KoopmanAutoencoder): The Koopman autoencoder
        dataset (tf.Tensor): The dataset to train the model on.
        checkpoint_path (str): The path to save the model weights.
    """

    # Default training settings
    max_epoch = train_settings.get('max_epoch', 1000)
    tol = train_settings.get('tol', 1e-3)
    alpha = train_settings.get('alpha', [1, 1, 0.3, 1e-9, 1e-9, 1e-9])
    optimizer = train_settings.get('optimizer', tf.keras.optimizers.Adam(learning_rate=1e-3))

    # Load and train model from checkpoint
    if checkpoint_path is not None:
        model, epoch = load_model(model, path=load_chekpoint_path)
        train(model=model, 
              dataset=train_dataset, 
              checkpoint_path=checkpoint_path, 
              epoch=epoch, 
              max_epoch=max_epoch, 
              tol=tol, 
              alpha=alpha, 
              optimizer=optimizer, 
              device="/GPU:0")
        
    # Train model from scratch
    else:
        epoch = 1
        train(model=model, 
              dataset=train_dataset, 
              checkpoint_path=checkpoint_path, 
              epoch=epoch, 
              max_epoch=max_epoch, 
              tol=tol, 
              alpha=alpha, 
              optimizer=optimizer, 
              device="/GPU:0")

def train(model: KoopmanAutoencoder, 
          dataset: tf.Tensor, 
          checkpoint_path: str, 
          epoch: int, 
          max_epoch: int, 
          tol: float, 
          alpha: list, 
          optimizer, 
          device: str):
    
    """
    Train the model with the given dataset and settings.

    Args:
        model (KoopmanAutoencoder): The Koopman autoencoder
        dataset (tf.Tensor): The dataset to train the model on.
        checkpoint_path (str): The path to save the model weights.
        epoch (int): The current epoch.
        max_epoch (int): The maximum number of epochs to train.
        tol (float): The tolerance for the loss.
        alpha (list): The weights for the loss function.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training.
        device (str): The device to use for training (e.g., "/GPU:0").
    """

    # Start training
    start = time.perf_counter()
    loss = 1
    prediction_steps = 50
    dataset_config = {'batch_size': 128, 'shuffle': True, 'sequence_length': prediction_steps + 1} 
    
    while (epoch <= max_epoch and tol <= loss):

        # Generate a batch over epoch
        train_batch = generate_train_batch(dataset, **dataset_config)
        
        # Train the model '/physical_device:GPU:0' '/gpu:0'
        model, loss = batch_train(model, train_batch, alpha, optimizer, device)
        tf.print(f"Epoch {epoch}/{max_epoch}, Loss =", loss)
        
        if epoch % 100 == 0:
            ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch}.weights.h5')
            model.save_weights(ckpt_path)

        # Timer 
        elapsed = time.perf_counter() - start
        print(f"Total training time: {elapsed:.2f} seconds")
        print("=====================================================")
        
        epoch += 1