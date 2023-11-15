
from pydantic import BaseModel


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
    id: int
        id of the wine
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
    id: int

    class Config:
        orm_mode = True
