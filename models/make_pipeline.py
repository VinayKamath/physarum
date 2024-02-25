from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys
import yaml

sys.path.insert(1, 'C://Users/vinay/Projects 2024/project_1/config')

from transformers import create_transformer

parameters = yaml.safe_load(open('../config/parameters.yml'))

r_state = parameters['train_test_split']['random_state']

def create_pipeline():
    preprocessor = create_transformer()

    model = RandomForestClassifier(random_state=r_state)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline
