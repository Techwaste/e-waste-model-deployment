from fastapi import FastAPI, File, UploadFile

import io
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import image as tfi

model = tf.keras.models.load_model('model/xception_latest.h5')
app = FastAPI()
class_names = ['battery', 'cable', 'crt_tv', 'e_kettle', 'fridge', 'keyboard',
               'laptop', 'light_bulb', 'monitor', 'mouse', 'pcb',
               'phone', 'printer', 'rice_cooker', 'washing_machine']

def preprocess_image(image):
    image = image.resize((300, 300))
    image_array = np.array(image)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    # Load the image
    # image = tf.io.read_file(image)
    # try:
    #     image = tfi.decode_jpeg(image, channels=3)
    # except:
    #     image = tfi.decode_png(image, channels=3)
    
    # # Change the image data type
    # image = tfi.convert_image_dtype(image, tf.float32)
    
    # # Resize the Image
    # image = tfi.resize(image, (300, 300))
    
    # # Convert image data type to tf.float32
    # convert_img = tf.cast(image, tf.float32)
    
    # resized_image = np.reshape(convert_img, (300, 300, 3))
    
    # tf_image = resized_image[tf.newaxis, ...]
    
    return image_array
    # return tf_image

def predict_image(image):
    predictions = model.predict(image)
    prediction_idx = np.argmax(predictions)
    
    predicted_class_name = class_names[prediction_idx]
    predicted_probability = np.max(predictions) * 100
    
    return predicted_class_name, predicted_probability

@app.get("/")
def read_root():
    return {"Welcome to TechWas (Technology Waste), Daniel"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # image = tf.io.read_file(file)
    image = Image.open(io.BytesIO(contents))
    tf_image = preprocess_image(image)
    # return image
    predict_class, predict_probability = predict_image(tf_image)
    
    return f"{predict_class} : {predict_probability}"
    