import pymongo
import csv

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Create database and collection
db = client["bacchus"]
collection = db["wines"]

# Check if collection already exists
if "wines" in db.list_collection_names():
    print("Collection already exists.")
else:
    # Read data from CSV file
    with open('Wines.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        wines = []
        for row in reader:
            wines.append(row)

    # Insert data into collection
    collection.insert_many(wines)
    print("Collection created and data inserted successfully.")
