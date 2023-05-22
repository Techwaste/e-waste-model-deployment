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

def predict_image(image: np.ndarray) -> tuple[str, float]:
    """Predicts the class name and probability for an image.

    Args:
        image (np.ndarray): The image to be predicted.

    Returns:
        tuple[str, float]: A tuple containing the predicted class name and probability.
    """
    predictions = model.predict(image)
    prediction_idx = np.argmax(predictions)
    
    predicted_class_name = class_names[prediction_idx]
    predicted_probability = np.max(predictions) * 100
    
    return predicted_class_name, predicted_probability