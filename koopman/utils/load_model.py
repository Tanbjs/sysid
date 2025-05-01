from koopman.models.deepkoopman import KoopmanAutoencoder

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
    load_model_status = True
    print("✅ Model weights loaded successfully.")
    epoch = int(path.split('_')[1].split('.')[0])  # Extract epoch from the filename
    print(f"✅ Model loaded at epoch {epoch}.")
    print("=====================================================")

    return model, epoch, load_model_status