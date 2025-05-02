from mypkg.koopman.models.deepkoopman import KoopmanAutoencoder

def load_model(model: KoopmanAutoencoder, path: str):
    """
    Load the model weights from a checkpoint file.
    
    Args:
        model (KoopmanAutoencoder): The model instance to load weights into.
    
    Returns:
        None
    """
    # Load the pretrained model weights
    model.load_weights(path)
    epoch = int(path.split('_')[1].split('.')[0])  # Extract epoch from the filename

    print("✅ Model weights loaded successfully.")
    print(f"✅ Model loaded at epoch {epoch}.")
    print("=====================================================")

    return model, epoch