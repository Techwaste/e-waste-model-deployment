import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model/xception_latest.h5')
class_names = ['Battery', 'Cable', 'CRT TV', 'E-kettle', 'Refrigerator',
                'Keyboard', 'Laptop', 'Light Bulb', 'Monitor', 'Mouse',
                'PCB', 'Phone', 'Printer', 'Rice Cooker', 'Washing Machine']

def preprocess_image(image: Image) -> np.ndarray:
    """Preprocesses an image for prediction.

    Args:
        image (PIL.Image): The image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    image = image.resize((300, 300))
    image_array = np.array(image)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(image: np.ndarray) -> dict:
    """Predicts the top 3 classes for an image using a pre-trained model.

    Args:
        image (np.ndarray): An image represented as a NumPy array.

    Returns:
        dict: A dictionary containing the top 3 predicted classes and their probabilities.
    """
    predictions = model.predict(image)
    top_3_indices = np.argsort(predictions)[0,-3:][::-1]

    data_predictions = {}
    for i in top_3_indices:
        data_predictions[class_names[i]] = predictions[0, i] * 100
        
    return data_predictions