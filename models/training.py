import sys
import yaml
import pickle

sys.path.insert(1, 'C://Users/vinay/Projects 2024/project_1/config')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from make_pipeline import create_pipeline
import pandas as pd

parameters = yaml.safe_load(open('../config/parameters.yml'))

dataset = parameters['dataset']['processed_data']
t_size = parameters['train_test_split']['test_size']
r_state = parameters['train_test_split']['random_state']

def train_model():
    df = pd.read_csv(dataset)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=r_state)

    pipeline = create_pipeline()

    pipeline = pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    
    pickle_out = open('pipe.pkl', 'wb')
    pickle.dump(pipeline, pickle_out)
    pickle_out.close()
    
    return pipeline
