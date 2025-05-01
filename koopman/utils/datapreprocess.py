import pandas as pd
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

def generate_train_batch(raw_data: tf.Tensor, **option) -> list:
    dataset = keras.utils.timeseries_dataset_from_array(data=raw_data, targets=None, **option)
    data = list(dataset.as_numpy_iterator())

    return data

def tfds2numpy(dataset: tf.data.Dataset) -> np.array:
    """
    Convert a TensorFlow dataset to a NumPy array.

    Args:
        dataset (tf.data.Dataset): TensorFlow dataset.

    Returns:
        np.array: NumPy array containing the data.
    """
    # Convert the dataset to a NumPy array
    numpy_array = np.array(list(dataset.as_numpy_iterator()))

    return numpy_array

def csv2tfds(file_path: str) -> tf.data.Dataset:
    """
    Convert a .csv file to a TensorFlow tensor.

    Args:
        file_path (str): Path to the .csc file.
        **option: Additional options for data processing.

    Returns:
        tf.Tensor: TensorFlow tensor containing the data.
    """
    # Read the .csc file
    raw_data = pd.read_csv(file_path)
    
    # Convert DataFrame to TensorFlow tensor
    dataset = tf.data.Dataset.from_tensor_slices(raw_data)

    return dataset

def tfds2tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Converts a tf.data.Dataset to a tf.Tensor with the original shape.
    Works for both batched and unbatched datasets.
    """
     # Convert the dataset to a list of tensors
    data_list = list(dataset.as_numpy_iterator())

    # Stack them into a single tensor
    return tf.convert_to_tensor(data_list)

def df2tfds(df: pd.DataFrame) -> tf.data.Dataset:
    """
    Convert a pandas DataFrame to a TensorFlow dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    # Convert DataFrame to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((df.values))

    return dataset