# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:48:52 2019
PyTest objects for the read_in_data functionality. 
@author: matt_
"""
import pytest
import sys
sys.path.append('.')
sys.path.append('./src')
from urllib.request import pathname2url
from src import read_in_data
import os
import pandas as pd

def test_short_datafile():
    """ Tests that loading process still produces expected data frame with 
    clean input
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_short_real_data.zip'.format(path))
    print(path)
    df = read_in_data.read_in_data(path)   
    df_check = pd.read_pickle("./test/test_data/first_row.pkl")

    assert all(df.head(1) == df_check)   
    assert all(df.head(1).keys() == df_check.keys()) 
    assert all(df.head(1).dtypes == df_check.dtypes) 

def test_empty_datafile():
    """ Tests empty datafile and makes sure an exception is raised. 
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_empty_data.zip'.format(path))
    print(path)
    with pytest.raises(Exception) as e_info:
        df = read_in_data.read_in_data(path)   
    assert e_info.value.args[0] ==("Empty file!"), "Empty file, check that source is not empty and that wasn't full of nulls"

def test_has_nulls():
    """ Checks that files with nulls are cleaned.
    For the scope of this project I am not going to extend it out to checking 
    for too many null rows being dropped etc.. 
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_has_nulls.zip'.format(path))
    df = read_in_data.read_in_data(path)   
    assert not df.isnull().any().any() , "The cleaned data still has nulls"


def test_has_extra_columns():
    """ Extra columns shouldn't cause any problems in loading. 
    Does some checks jsut to make sure the data loads
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_extra_column.zip'.format(path))
    df = read_in_data.read_in_data(path)   
    df_check = pd.read_pickle("./test/test_data/first_row.pkl")
    assert all(df[df_check.keys()].head(1) == df_check)   
    assert all([key in df.head(1).keys() for key in df_check.keys()]) 
    assert all(df[df_check.keys()].head(1).dtypes == df_check.dtypes) 

def test_has_missing_columns():
    """ Missing Columns should raise error
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/Bike-Sharing-Dataset_missing_column.zip'.format(path))
    with pytest.raises(Exception) as e_info:
        df = read_in_data.read_in_data(path) 
    assert e_info.value.args[0] == "Column missing from original training data, check data source", "Missing Key hasn't raised KeyError as it should"
 
def test_no_file():
    """ Missing Columns should raise error
    """
    path = os.getcwd()
    path = r'file:' + pathname2url(r'{}/test/test_data/asantasanasquashedbanana.zip'.format(path))
    with pytest.raises(IOError) as e_info:
        df = read_in_data.read_in_data(path) 
