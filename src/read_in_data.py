# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:07:20 2019
Opens and preprocesses data. 
@author: matt_
"""
import pandas as pd
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
#import config


def read_in_data(training_data_url, file = "hour.csv"):
    """Grabs the data from the UCI website.
    File is a zip file with three files. We only load the hour csv file. 
    Loads it into a dataframe. 
    """       
    try:
        zipped = urlopen(training_data_url)
        myzip = ZipFile(BytesIO(zipped.read())).extract(file)
    except AttributeError:
        print("Check the URL in cofig currently inputting: {}".format(training_data_url)) 
        raise
    try:
        df = pd.read_csv(myzip)
    except IOError:
        print("Reading the pandas file didn't work.: {}".format(file)) 
    print(df.head())
    #Just drop rows with nans
    df.dropna(inplace=True)
    
    assert df.shape[0]>0 , "Empty file!"

    # Check all columns present    
    expected_keys = ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt']
    assert all([key in df.keys() for key in expected_keys]), "Column missing from original training data, check data source"
    
    return df


if __name__ == "__main__":
    
    df = read_in_data(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip')
    
