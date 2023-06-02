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
import requests
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

def getID(name1,name2,name3):
  mydb = mysql.connector.connect(
  host="34.69.199.102",
  user="root",
  password="J]91kx6G&S:^]'Gu",
  database="components"
  )
  mycursor = mydb.cursor()
  mycursor.execute("SELECT id FROM comps WHERE name = %s OR name = %s OR name = %s ORDER BY id DESC",(name1,name2,name3,))
  myresult = mycursor.fetchall()
  meow={
    name1:str(myresult[0]).replace("(", "").replace(")", "").replace(",", ""),
    name2:str(myresult[1]).replace("(", "").replace(")", "").replace(",", ""),
    name3:str(myresult[2]).replace("(", "").replace(")", "").replace(",", "")
  }
  return meow



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
    listed = list(data_predict.keys())
    getID(listed[0],listed[1],listed[2])
    
    return {
            "predictions": data_predict,
            "id":getID(listed[0],listed[1],listed[2],),
            "time_taken": f.timer(time),
            }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
