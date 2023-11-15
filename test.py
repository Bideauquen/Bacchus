import requests

def test_put_wine():
    wine = {
        "fixed_acidity": 7.32,
        "volatile_acidity": 0.65,
        "citric_acid": 0,
        "residual_sugar": 1.7,
        "chlorides": 0.081,
        "free_sulfur_dioxide": 12,
        "total_sulfur_dioxide": 33,
        "density": 0.9978,
        "pH": 3.49,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "quality": 6
    }
    response = requests.put("http://0.0.0.0:8080/api/model", json=wine)
    assert response.status_code == 200

if __name__ == "__main__":
    test_put_wine()