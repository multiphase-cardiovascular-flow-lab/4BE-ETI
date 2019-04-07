import logging
import math
import os
import re

import numpy as np
import pandas as pd

## This file introduces functions that are needed

def initialize_logfile(main_dir, filename):
    """
    Rounds a number up to the nearest multiple of the value defined by increment
    Input: main_dir - directory that contains the results (tracking, plotting, etc.)
           filename - specified name for logfile
    Output: initialized logfile
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(main_dir + filename)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def roundup(x, increment):
    """
    Rounds a number up to the nearest multiple of the value defined by increment
    Input: x - number to round up to the nearest multiple value
           increment - number to use to round up to (ie. if x is 225 and increment
                       is 50, then the output of the function would be 250)
    Output: Rounded value of x
    """
    return int(math.ceil(x / increment)) * increment


# Orders files sequentially
# (ie. 1, 2, 3, 4)
def natural_key(string_):
    '''See http://www.codinghorror.com/blog/archives/001018.html'''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)',
            string_)]


class a:
    """
    Defines a class named a to label the data more easily. For example, this
    class allows data['X'] to be redefined as data.x
    """
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.Area = []
        self.Slice = []
        self.Count = []
        self.CountTemp = []
        self.Cost = []


def convert_class(data_df):
    """
    Changes the way that data is stored and also adds the columns for count,
    tempcount, and cost
    Input: data_df - pandas df containing the particle data
    Output: data - class data
    """
    data = a()
    data.x = data_df['X'].values
    data.y = data_df['Y'].values
    if 'Z' in data_df:
        data.z = data_df['Z'].values
    data.Area = data_df['Area'].values
    data.Slice = data_df['Slice'].values.astype('int')
    data.Count = data_df['Count'].values.astype('int')
    data.CountTemp = data_df['CountTemp'].values.astype('int')
    data.Cost = data_df['Cost'].values.astype('float')

    return data


def load_data(main_dir, trial_name, file_type):
    """
    Loads data from a file into a numpy array to use with the tracking program
    Input: main_dir - directory that contains the particle data
    Output: data - data class containing the results
    """
    
    if file_type == '.csv':
        data_df = pd.read_csv(main_dir + trial_name + file_type)
    elif file_type == '.txt':
        data_df = pd.read_csv(main_dir + trial_name + file_type, 
                              sep=None, engine='python')
    elif file_type == '.pkl':
        data_df = pd.read_pickle(main_dir + trial_name + file_type)
    else:
        raise ValueError('Particle file must be either a csv, txt, or pkl file')

    if 'Area' not in data_df:
        data_df['Area'] = -np.ones(len(data_df)) 
    
    data_df['Count'] = np.zeros(len(data_df))
    data_df['CountTemp'] = np.zeros(len(data_df))
    data_df['Cost'] = -np.ones(len(data_df))

    data = convert_class(data_df)

    return data
