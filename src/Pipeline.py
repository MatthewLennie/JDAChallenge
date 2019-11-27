# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:01:41 2019

@author: matt_
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


class Pipeline():
    
    def __init__(config_dict):
        self.config_dict = config_dict
        
    def create_pipeline():
        pass
    

    def create_test_training_sets(df):
        #%%
        #df = df
        enc = pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(df['weekday'].values.reshape(-1,1)))
        df = pd.concat([df, enc],axis=1)

        #%%
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df[self.x_keys].astype("float"), df[self.y_keys].astype("float"), test_size=0.33, random_state=42)

    
    def hyper_parameter_search():
        pass
    
    def train_with_optimal_model():
        pass
    
    def predict():
        pass
    
    def assess_model():
        pass


if __name__ == "__main__":
    
    df = read_in_data(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip')
            