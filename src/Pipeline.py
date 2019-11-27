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
import config
import sklearn.pipeline

class Pipeline():
    
    def __init__(self,config):
        assert self.__check_config_file
        self.config = config
        
    def create_pipeline():
        pass
    
    def __check_config_file(self):
        return False
    
    def __preprocess_features(self,df): 
        enc = pd.DataFrame(OneHotEncoder(sparse=False,categories='auto').fit_transform(df['weekday'].values.reshape(-1,1)))
        df = pd.concat([df, enc],axis=1)
        self.config.x_keys.extend(enc.keys()[:-1]) #don't over determine the system
        self.config.x_keys = [x for x in self.config.x_keys if x!='weekday']
        return df
    
    def create_test_training_sets(self,df):
        """Encodes the weekday and performs test train split. 
        """
        #turn weekdays into one hot encode remove original feature
        
        #%%
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df[self.config.x_keys].astype("float"), df[self.config.y_keys].astype("float"), test_size=0.33, random_state=self.config.random_seed)

    def load_in_pretrained():
        pass
    
    def hyper_parameter_search(self):
        #To speed up set to DASK backend...
        clf = xgb.XGBRegressor(n_jobs=self.config.n_jobs)
        search = GridSearchCV(clf,self.config.parameters,cv=self.config.number_of_cross_validations)
        search.fit(self.X_train,self.y_train)
        self.model = search.best_estimator_
    
    def build_pipeline(self):
        self.pipeline = sklearn.pipeline.Pipeline([('anova', anova_filter), ('svc', clf)])
    
    def predict(self,x_data):
        
    
    def assess_model():
        pass


if __name__ == "__main__":
    
    df = read_in_data(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip')
    example_pipe = Pipeline(config)
    example_pipe.create_test_training_sets(df)
    
    
    
    
    
    
    
    
    
    