from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import pathlib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local', force_reload=True)

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

checklist = ["map", "rope", "knife", "candle", "lighter", "radio", "ID_card", "flashlight", "matches", 
             "chocolate_bar", "canned_tuna", "water", "water_purification_tablets", "candy", "instant_noodles", "heat_pack", 
             "solid_fuel", "umbrella", "towel", "blanket", "backpack", "sleeping_bag", "whistle", 
             "ziplock_bag", "toothbrush", "mask", "tissue", "first_aid_kits", "thermal_blanket"]

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        result = model(image)
        result.show()

        recognized_items = set(result.names[int(pred[5])] for pred in result.xywh[0])

        missing_items = [item for item in checklist if item not in recognized_items]

        return JSONResponse(content={
            "status": "success",
            "recognized_items": list(recognized_items),
            "missing_items": missing_items
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
