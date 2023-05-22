from fastapi import File, UploadFile

import io
from PIL import Image
import os
import uvicorn
import api_config as config
import func as f
from api_config import Tags

port = int(os.getenv("PORT"))

app = config.app

@app.get("/")
def read_root():
    return {"Welcome to TechWas (Technology Waste)"}

@app.post("/predict/", tags=[Tags.predict])
async def predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg or jpeg format!"
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    tf_image = f.preprocess_image(image)
    data_predict = f.predict_image(tf_image)

    return data_predict

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)

