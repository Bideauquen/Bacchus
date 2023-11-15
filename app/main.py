
from fastapi import FastAPI

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