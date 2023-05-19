
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage

from PIL import Image
import numpy as np
import tensorflow as tf
import os
import uvicorn
import io
import random
import json
import mysql.connector

key_pi = os.getenv("cres")
GOOGLE_APPLICATION_CREDENTIALS = key_pi

form = str(random.randint(3, 3265792139879102375))
port = int(os.getenv("PORT"))
bucketName="techpybarahh"
model = tf.keras.models.load_model('model/xception_latest.h5')
app = FastAPI()
class_names = ['battery', 'cable', 'crt_tv', 'e_kettle', 'fridge', 'keyboard',
               'laptop', 'light_bulb', 'monitor', 'mouse', 'pcb',
               'phone', 'printer', 'rice_cooker', 'washing_machine']

def storage_thingy(blobName,filePath,bucketName):
    storageClient = storage.Client()
    dir(storageClient)
    bucket = storageClient.bucket(bucketName)
    vars(bucket)
    bucket = storageClient.get_bucket(bucketName)
    blob = bucket.blob(blobName)
    blob.upload_from_filename(filePath)

    return blob


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

def getCompId(name):
    
    mydb = mysql.connector.connect(
        host=os.getenv("HOST"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        database="components"
    )

    mycursor = mydb.cursor()
    meow = name
    meowmeow = (meow,)
    mycursor.execute("SELECT * FROM comps WHERE name= %s", meowmeow)
    myresult = mycursor.fetchall()
    compid = myresult[0][0]
    if compid is not None:
        return compid
    else:
        return False


@app.get("/")
def read_root():
    return {"Welcome to TechWas (Technology Waste), Daniel"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg, jpeg, or png format!"
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image.save(file.filename)
    tf_image = preprocess_image(image)
    predict_class, predict_probability = predict_image(tf_image)
    storage_thingy("predictSave/"+predict_class+form,file.filename,bucketName)
    os.remove(file.filename)
    theMeowMeow=getCompId(predict_class)
    return {
        "compName":theMeowMeow,
        "predict_class" : "predict_probability"
        }



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
