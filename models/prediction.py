import pandas as pd
from training import train_model

model = train_model()

d = {'Pclass': [2], 'Sex': 'male', 'Age': 31.0, 'Sibsp': 0, 'Parch': 0, 'Fare': 10.5, 'Embarked': 'S'}
df = pd.DataFrame(d)

prediction = model.predict(df)

print(prediction)
