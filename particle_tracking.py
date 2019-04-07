import basic_utils
import logging
import os
import time

import numpy as np
import pandas as pd

from particle_tracking_code import previous_tracks
from particle_tracking_code import previous_tracks_3d
from particle_tracking_code import no_previous_tracks
from particle_tracking_code import no_previous_tracks_3d


"""
Runs particle tracking code on a given file

Required user inputs are directly below
"""

# directory containing the file with particle information (location, area, frame number)
main_dir = 
# name of file containing the particle information
trial_name =
# type of file containing the particle information (options include: .txt, .csv, .pkl) 
file_type =

# 2d or 3d tracking?
dimension = 
# box size in x direction for track initialization (a good initial guess is the expected 
# maximum displacement of the particles in the x direction between frames)
box_size_initial_x =
# box size in y direction for track initialization 
box_size_initial_y = 
# box size in z direction for track initialization
box_size_initial_z =
# box size used after a track is initialized (this should be as small as possible to 
eliminate spurious track)
box_size =






"""
Start of the code; user should not change anything after this point 
"""


data = basic_utils.load_data(main_dir, trial_name, file_type) 

logger = basic_utils.initialize_logfile(main_dir, 'logfile_' + trial_name + '.log')

logging.info(main_dir)
logging.info(trial_name)
logging.info('Particle tracking initialized.')
    

start_time = time.time()

for jj in range(1, self.data.Slice.max() + 1):
    if jj % 500 == 0:
        logging.info(str(jj))
        logging.info(str(time.time() - start_time) + 'seconds')
    imInit = np.where(data.Slice == jj)[0]
    for ii in range(len(imInit)):
        if data.Count[imInit[ii]] == 0:
            if dimension == '3d':
                data = no_previous_tracks_3d(data, jj, ii, box_size, 
                                             box_size_initial_x,
                                             box_size_initial_y,
                                             box_size_initial_z)
            else:
                data = no_previous_tracks(data, jj, ii, box_size, 
                                          box_size_initial_x,
                                          box_size_initial_y)
        else:
            if dimension == '3d':
                data = previous_tracks_3d(data, jj, ii, box_size)
            else:
                data = previous_tracks(data, jj, ii, box_size)

    if len(data.Count[data.Count == -1]) > 0:
        data.Count[data.Count == (-1)] = 0

if dimension == '3d':
    data_final = pd.DataFrame({'X': data.x, 'Y': data.y, 'Z': data.z,
                               'Slice': data.Slice, 'Count': data.Count,
                               'Cost': data.Cost, 'Area': data.Area})      
else:
    data_final = pd.DataFrame({'X': data.x, 'Y': data.y, 
                               'Slice': data.Slice, 'Count': data.Count,
                               'Cost': data.Cost, 'Area': data.Area})

data_final.to_pickle(main_dir + 'tracking_results_box_size_' + trial_name + '.pkl')

logging.info('Particle tracking program took ' + str(time.time() - start_time) + ' seconds to run.')
