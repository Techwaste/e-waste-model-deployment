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

port = int(os.getenv("PORT"))

key_pi = os.getenv("cres")
GOOGLE_APPLICATION_CREDENTIALS = key_pi

app = config.app
form = str(random.randint(3, 3265792139879102375))
bucketName = "techpybarahh"


def storage_thingy(blobName, filePath, bucketName):
    storageClient = storage.Client()
    dir(storageClient)
    bucket = storageClient.bucket(bucketName)
    vars(bucket)
    bucket = storageClient.get_bucket(bucketName)
    blob = bucket.blob(blobName)
    blob.upload_from_filename(filePath)

    return blob


def getCompId(name):
    mydb = mysql.connector.connect(
        host="34.69.199.102",
        user="root",
        password="J]91kx6G&S:^]'Gu",
        database="components",
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
    return {"Welcome to TechWas (Technology Waste)"}


@app.post("/predict/", tags=[Tags.predict])
async def predict(file: UploadFile = File(...)):
    time = f.timer(None)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg or jpeg format!"
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    image.save(file.filename)
    tf_image = f.preprocess_image(image)
    data_predict = f.predict_image(tf_image)
    savedClass = data_predict.keys()
    savedClass = list(savedClass)
    savedClass = savedClass[0]
    storage_thingy("predictSave/" + savedClass + form, file.filename, bucketName)
    os.remove(file.filename)
    return {"predictions": data_predict, "time_taken": f.timer(time)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
