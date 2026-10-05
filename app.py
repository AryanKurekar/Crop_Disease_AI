import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER="static/uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

model=load_model("crop_disease_model.h5")

with open("class_indices.json") as f:
    class_indices=json.load(f)

classes={v:k for k,v in class_indices.items()}

disease_info = {

    "healthy": {
        "name": "Healthy Leaf",
        "causes": "No disease detected.",
        "prevention": "Maintain proper watering and sunlight.",
        "treatment": "No treatment required."
    },

    "early_blight": {
        "name": "Early Blight",
        "causes": "Fungal infection due to warm, humid conditions.",
        "prevention": "Avoid overhead watering, rotate crops.",
        "treatment": "Use fungicides like chlorothalonil."
    },

    "late_blight": {
        "name": "Late Blight",
        "causes": "Caused by Phytophthora infestans fungus.",
        "prevention": "Ensure good air circulation, avoid wet leaves.",
        "treatment": "Use copper-based fungicides."
    },

    "mosaic_virus": {
        "name": "Mosaic Virus",
        "causes": "Viral infection spread by insects.",
        "prevention": "Control insects like aphids.",
        "treatment": "Remove infected plants."
    }
}

@app.route("/",methods=["GET","POST"])
def index():

    prediction=None
    info=None
    image_path=None

    if request.method=="POST":

        file=request.files["file"]

        filepath=os.path.join(UPLOAD_FOLDER,file.filename)

        file.save(filepath)

        image_path=filepath

        img=Image.open(filepath).convert("RGB")
        img=img.resize((224,224))

        img=np.array(img)/255.0
        img=np.expand_dims(img,axis=0)

        pred=model.predict(img)

        index=np.argmax(pred)

        prediction=classes[index]

        info=disease_info[prediction]

    return render_template(
        "index.html",
        prediction=prediction,
        info=info,
        image_path=image_path
    )

if __name__=="__main__":
    app.run(debug=True)

