
from fastapi import FastAPI
import websockets

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from db.base import Wine
from db.mongo_cli import MongoDatabase

app = FastAPI()

db = MongoDatabase()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Put a wine in the database
@app.put("/api/model")
async def add_wine(wine : Wine):
    wine_dict = wine.model_dump()
    id = db.wines.insert_one(wine_dict)
    return {"message": "Wine {} added to database".format(id)}

# Get the serialized model
@app.get("/api/model")
async def get_model():
    return {"model": db.wines.find_one()}

# Retrain the model
@app.post("/api/model/retrain")
async def retrain_model():
    # Retrieve the data from the database
    data = db.wines.find()
    # Send the data to the model docker through a websocket
    async with websockets.connect("ws://model:8765") as websocket:
        await websocket.send(data)
        await websocket.recv()
