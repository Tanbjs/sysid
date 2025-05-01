# system module
import time 
import os
# Path to the main directory
import tensorflow as tf
# own package
from koopman.models.deepkoopman import KoopmanAutoencoder
from koopman.utils.batch_train import batch_train
from koopman.utils.datapreprocess import generate_train_batch

def train(model: KoopmanAutoencoder, dataset: tf.Tensor, checkpoint_path: str):

    # Training settings
    max_epoch = 1000
    tol = 1e-3
    alpha = [1, 1, 0.3, 1e-9, 1e-9, 1e-9]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Start training
    start = time.perf_counter()
    epoch = 1
    loss = 1
    prediction_steps = 50
    dataset_config = {'batch_size': 128, 'shuffle': True, 'sequence_length': prediction_steps} 
    
    while (epoch <= max_epoch and tol <= loss):

        # Generate a batch over epoch
        dataset = generate_train_batch(dataset, **dataset_config)
        
        # Train the model
        loss = batch_train(model, dataset, alpha, optimizer, device='/gpu:0')
        tf.print(f"Epoch {epoch}/{max_epoch}, Loss =", loss)
        
        if epoch % 100 == 0:
            ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch}.weights.h5')
            model.save_weights(ckpt_path)

        print("=====================================================")
        
        # Timer 
        elapsed = time.perf_counter() - start
        print(f"Total training time: {elapsed:.2f} seconds")
        
        epoch += 1