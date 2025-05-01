import numpy as np

def test(model, test_dataset: tuple):
    """
    Calculate and print the Mean Squared Error (MSE) and Maximum Absolute Error (MaxAE) of the model's predictions.
    
    Args:
        model: The trained TensorFlow model.
        test_set: The input tensor for testing.
        true_values: The ground truth tensor corresponding to the test_set.
    """
    # Get predictions from the model
    predictions = model(test_dataset, training=False)

    # Calculate prediction error
    prediction_error = test_dataset - predictions

    # Compute Mean Squared Error (MSE)
    mse = np.mean(np.square(prediction_error))

    # Compute Maximum Absolute Error (MaxAE)
    max_ae = np.max(np.abs(prediction_error))

    # Print the errors
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Maximum Absolute Error (MaxAE): {max_ae}")

    return predictions
