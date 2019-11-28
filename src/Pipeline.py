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
import pickle
class Preprocess():
    """ Simple wrapper for the encoder for the weekdays. 
    If you want to implement normalization etc.. here would be the spot. 
    """
    def __init__(self,config):
        self.config = config
        
    def fit(self,df):
        self.encoder = OneHotEncoder(sparse=False,categories='auto').fit(df['weekday'].values.reshape(-1,1))
        
        self.config.x_keys.extend([0,1,2,3,4,5]) #don't over determine the system
        
    def transform(self,df):
        day_encoded = pd.DataFrame(self.encoder.transform(df['weekday'].values.reshape(-1,1)))   
        df = pd.concat([df, day_encoded],axis=1)
        if 'cnt' in df.keys():
            self.y_data = df['cnt'].astype('float')
            
        #remove weekday from the keys. 
        self.config.x_keys = [x for x in self.config.x_keys if x!='weekday']
        
        return df[self.config.x_keys].astype('float')
    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)
    
    
class Pipeline():    
    def __init__(self,config):
        assert self.__check_config_file
        self.config = config
        
    def create_pipeline():
        pass
    
    def __check_config_file(self):
        return False
    
    
    def create_test_training_sets(self,df):
        """Encodes the weekday and performs test train split. 
        """
        #turn weekdays into one hot encode remove original feature
        self.preprocessor = Preprocess(self.config)
        
        x = self.preprocessor.fit_transform(df)
        y = self.preprocessor.y_data
        
        self.config = self.preprocessor.config

        #%%
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.33, random_state=self.config.random_seed)

    
    def hyper_parameter_search(self):
        """performs a search of hyper-parameters on data. 
        """
        try:
            clf = xgb.XGBRegressor(n_jobs=self.config.n_jobs)
            search = GridSearchCV(clf,self.config.parameters,cv=self.config.number_of_cross_validations)
            search.fit(self.X_train,self.y_train)
            self.model = search.best_estimator_
        except AttributeError:    
            print("Have you created training data?")
            
    
    def build_pipeline(self):
        """ Packs the process into a sklearn pipeline ready for inference. 
        """
        try:
            self.pipeline = sklearn.pipeline.Pipeline([('preprocessor', self.preprocessor), ('XGmodel', self.model)])
        except AttributeError:
            print("the model hasn't been trained or loaded")
        
    def pickle_pipeline(self,filename = "bicycle_tree.pkl"):
        """pickles pipeline 
        """
        pickle.dump(self.model,filename)
        
    def unpickle_pipeline(self,filename = "bicycle_tree.pkl"):
        """pickles pipeline 
        """
        self.model =pickle.load(filename)
        
    def predict(self,df):
        """Easy access to the predict function. 
        """
        self.pipeline.predict(df)
    
    def retrain_model(self,df):
        """Takes the model with the optimized hyperparameters and refits with
        new data.
        """
        #TODO
        raise NotImplementedError
    
    def assess_model():
        pass


if __name__ == "__main__":
    
    df = read_in_data(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip')
    example_pipe = Pipeline(config)
    example_pipe.create_test_training_sets(df)
    example_pipe.hyper_parameter_search()
    example_pipe.build_pipeline()
    
    
    example_pipe.pipeline.predict(df)
#    example_pipe.pipeline.(df)
    
    
    
    
    