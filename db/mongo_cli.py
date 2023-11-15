import pymongo
import csv

### Utils ###

def get_config_from_file(config_file : str):
        # Read config file
        with open(config_file, 'r') as f:
            config = f.readlines()
            # Port is on the second line
            port = int(config[1].split(":")[1].strip())
            # bind_ip is on the third line
            bind_ip = config[2].split(":")[1].strip()
        return port, bind_ip

def get_credentials_from_file(credentials_file : str):
    # Read credentials file
    with open(credentials_file, 'r') as f:
        credentials = f.readlines()
        for line in credentials:
            if "USERNAME" in line:
                username = line.split("=")[1].strip()
            elif "PASSWORD" in line:
                password = line.split("=")[1].strip()
    return username, password


### MongoDatabase class ###

class MongoDatabase:
    def __init__(self, config_file : str = "db/mongod.conf", credentials_file : str = "db/mongo-auth.env"):

        """
        Initializes a MongoDB connection and sets up the database and collections.
        
        Parameters
        ----------
        config_file : str
            The path to the MongoDB configuration file.
        credentials_file : str
            The path to the MongoDB credentials file.
        
        Attributes
        ----------
        client : pymongo.MongoClient
            The MongoDB client.
        db : pymongo.database.Database
            The MongoDB database named "bacchus".
        wines : pymongo.collection.Collection
            The MongoDB collection for wines, will contain wine characteristics (cf Wine class)
        """

        port, _ = get_config_from_file(config_file)
        username, password = get_credentials_from_file(credentials_file)
        self.client = pymongo.MongoClient(f"mongodb://{username}:{password}@mongo:{port}/")
        self.db = self.client["bacchus"]
        self.wines = self.db["wines"]
        self.models = self.db["models"]

        # If the wines collection is empty, insert data from csv file
        if self.wines.count_documents({}) == 0:
            csv_reader = CsvReader("db/Wines.csv")
            data = csv_reader.read_csv()
            # Remove the id column
            for row in data:
                del row["Id"]
            self.insert_data("wines", data)

    def create_collection(self, collection_name):
        if collection_name in self.db.list_collection_names():
            print("Collection already exists.")
        else:
            self.db.create_collection(collection_name)
            print("Collection created successfully.")

    def insert_data(self, collection_name, data):
        collection = self.db[collection_name]
        collection.insert_many(data)
        print("Data inserted successfully.")

    def get_last_model(self):
        """
        Returns the last model in the database (using the version number).
        
        Returns
        -------
        dict
            The last model in the database.
        """
        return self.models.find_one(sort=[("version", pymongo.DESCENDING)])

class CsvReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_csv(self):
        with open(self.file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = []
            for row in reader:
                data.append(row)
        return data

if __name__ == '__main__':
    MongoDatabase()
