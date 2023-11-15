
from pydantic import BaseModel
from bson import ObjectId


# Here are the attributes of the Wine class
# fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id


class Wine(BaseModel):

    """
    This class is used to create a pydantic model of the Wine class
    
    Attributes:
    -----------
    fixed_acidity: float
        fixed acidity of the wine
    volatile_acidity: float
        volatile acidity of the wine
    citric_acid: float
        citric acid of the wine
    residual_sugar: float
        residual sugar of the wine
    chlorides: float
        chlorides of the wine
    free_sulfur_dioxide: float
        free sulfur dioxide of the wine
    total_sulfur_dioxide: float
        total sulfur dioxide of the wine
    density: float
        density of the wine
    pH: float
        pH of the wine
    sulphates: float    
        sulphates of the wine
    alcohol: float
        alcohol of the wine
    quality: float
        quality of the wine
    """
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: float

class Layer(BaseModel) :
    """
    This class is used to create a pydantic model of the Layer class

    Attributes:
    -----------
    type: str
        type of the layer
    units: int
        number of units of the layer
    activation: str
        activation function of the layer
    """
    type: str
    units: int
    activation: str

class Structure(BaseModel) :
    """
    This class is used to create a pydantic model of the Structur class

    Attributes:
    -----------
    layers: list
        list of layers of the model
    """

    layers: list[Layer]
    nb_params: int

class Model(BaseModel) :
    """
    This class is used to create a pydantic model of the Model class

    Attributes:
    -----------
    version: str
        version of the model
    type: str
        type of the model
    structure: dict
        structure of the model
    model : dict
        serialized model
    """
    version: str
    type: str
    structure: Structure
    model : dict

