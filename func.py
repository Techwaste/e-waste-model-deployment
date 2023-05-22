import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/xception_latest.h5')
# class_names = ['battery', 'cable', 'crt_tv', 'e_kettle', 'fridge', 'keyboard',
#                'laptop', 'light_bulb', 'monitor', 'mouse', 'pcb',
#                'phone', 'printer', 'rice_cooker', 'washing_machine']
class_names = ['Baterai', 'Kabel', 'CRT TV', 'E-kettle', 'Kulkas',
               'Keyboard', 'Laptop', 'Bolam', 'Monitor', 'Mouse',
               'PCB', 'Phone', 'Printer', 'Rice Cooker', 'Mesin Cuci']

def preprocess_image(image):
    image = image.resize((300, 300))
    image_array = np.array(image)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(image):
    predictions = model.predict(image)
    prediction_idx = np.argmax(predictions)
    
    predicted_class_name = class_names[prediction_idx]
    predicted_probability = np.max(predictions) * 100
    
    return predicted_class_name, predicted_probability

# def convert_png_to_jpg(image):
#     png_image = tf.io.read_file(image)
#     png_image = tf.image.decode_png(png_image)
#     png_image = tf.cast(png_image, tf.uint8)
#     png_image = tf.image.encode_jpeg(png_image)
    
#     # image = image.convert('RGB')
#     # jpeg_data = io.BytesIO()
#     # image.save(jpeg_data, format='JPEG')
    
#     # return jpeg_data.getvalue()
    
#     return png_image