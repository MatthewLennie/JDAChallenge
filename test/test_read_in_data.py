# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:48:52 2019

@author: matt_
"""
import pytest
import sys
sys.path.append('.')
sys.path.append('./src')
from urllib.request import pathname2url
from src import read_in_data
import os
def test_short_datafile():
#    test_url= r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
    path = os.getcwd()
#    print('file://{}\test\test_data\Bike-Sharing-Dataset_short_real_data.zip'.format(path))
    
#    path = r'file:' + pathname2url(r'C:/Users/matt_/Documents/JDAChallenge/test/test_data/Bike-Sharing-Dataset_short_real_data.zip')
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_short_real_data.zip'.format(path))
    print(path)
    read_in_data.read_in_data(path)   
    raise
    