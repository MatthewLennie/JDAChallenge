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


def read_in_data(training_data_url):
    """Grabs the data from the UCI website.
    File is a zip file with three files. We only load the hour csv file. 
    Loads it into a dataframe. 
    """       
    try:
        zipped = urlopen(training_data_url)
        myzip = ZipFile(BytesIO(zipped.read())).extract('hour.csv')
    except AttributeError:
        print("Check the URL in cofig currently inputting: {}".format(training_data_url)) 
        raise
    df = pd.read_csv(myzip)
    return df
