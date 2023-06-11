from fastapi import File, UploadFile
from google.cloud import storage
import io
from PIL import Image
import os
import uvicorn
import api_config as config
import func as f
from api_config import Tags
import random
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

print(os.getenv("cres"))
port = int(os.getenv("PORT"))

key_pi = os.getenv("cres")
# GOOGLE_APPLICATION_CREDENTIALS = key_pi
with open("service_account.json", "w") as file:
    file.write(key_pi)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_account.json"
app = config.app
form = str(random.randint(3, 3265792139879102375))
bucketName = "techwaste"



def storage_thingy(blobName, filePath, bucketName):
    storageClient = storage.Client()
    dir(storageClient)
    bucket = storageClient.bucket(bucketName)
    vars(bucket)
    bucket = storageClient.get_bucket(bucketName)
    blob = bucket.blob(blobName)
    blob.upload_from_filename(filePath)

    return blob


@app.get("/")
def read_root():
    return {"Welcome to TechWas (Technology Waste)"}


@app.post("/predict/", tags=[Tags.predict])
async def predict(file: UploadFile = File(...)):
    time = f.timer(None)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg or jpeg format!"
    contents = await file.read()
    savedform = form
    image = Image.open(io.BytesIO(contents))
    image.save(file.filename)
    tf_image = f.preprocess_image(image)
    data_predict = f.predict_image(tf_image)
    savedClass = data_predict
    savedClass = str(savedClass[0]["Components Name"])
    storage_thingy("predictSave/" + savedClass + savedform, file.filename, bucketName)
    os.remove(file.filename)
    return {"predictions": data_predict,
     "time_taken": f.timer(time),
     "Image_Url": "https://storage.googleapis.com/techwaste/predictSave/"+savedClass+savedform
     }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
