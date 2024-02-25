from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from training import train_model
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to my ml prediction': f'{name}'}

@app.post("/predict")
def predict(item: Item):
    try:
        model = train_model()
        df = pd.DataFrame({
            'Pclass': [item.Pclass],
            'Sex': [item.Sex],
            'Age': [item.Age],
            'Sibsp': [item.SibSp],
            'Parch': [item.Parch],
            'Fare': [item.Fare],
            'Embarked': [item.Embarked]
        })
        prediction = model.predict(df)
        
        # if int(prediction[0]) == 0:
        #     prediction_result = "The passenger didn't survive"
        # else:
        #     prediction_result = "The passenger survived"

        # return {
        #     "prediction": int(prediction[0]),
        #     "result": prediction_result
        # }
    
        if int(prediction[0]) == 0:
            prediction = "The passenger didn't survive"
        else:
            prediction = "The passenger survived"
        return {
            "prediction": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




    

