# system module
import time 
import os
# Path to the main directory
import tensorflow as tf
# own package
from koopman.models.deepkoopman import KoopmanAutoencoder
from koopman.utils.batch_train import batch_train
from koopman.utils.datapreprocess import generate_train_batch

def train(model: KoopmanAutoencoder, dataset: tf.Tensor, checkpoint_path: str, 
          epoch: int, max_epoch: int = 1000, tol: float = 1e-3, alpha: list = None, 
          optimizer: tf.keras.optimizers.Optimizer = None, device: str = None):

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