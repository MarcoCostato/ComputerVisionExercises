from app.services.vision.baseClassifier import BaseClassifier
import xgboost as xgb
import numpy as np

class  XGBoostClassifier(BaseClassifier):

    def __init__(self):
        self.model = None

    def load(self, model_path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def predict(self, features):
        return self.model.predict(features)