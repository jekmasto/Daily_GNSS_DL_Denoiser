#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Jul  20 15:52:21 2023
@author: giacomo

A simple code to download GNSS daily displacement time-series from the Nevada repository [http://geodesy.unr.edu]

"""

import pandas as pd
import numpy as np
import datetime, os,requests
import pandas as pd

def import_stations(saving_cd_coord,lat_south,lat_north,lon_west,lon_east):

    """
    Parameters
    ----------
        saving_cd_coord: where save the coordinates file
        lat_south,lat_north,lon_west,lon_east: boundaries of coordinates

    Returns
        
    """
    
    if not os.path.exists(saving_cd_coord):
        os.makedirs(saving_cd_coord)
        print(f"Folder '{saving_cd_coord}' created.")
    else:
        print(f"Folder '{saving_cd_coord}' already exists.")
    
    saving_cd=saving_cd_coord+'/Stations'
    if not os.path.exists(saving_cd):
        os.makedirs(saving_cd)
        print(f"Folder '{saving_cd}' created.")
    else:
        print(f"Folder '{saving_cd}' already exists.")
        
    
    ### All stations ###
    url_stations = "http://geodesy.unr.edu/NGLStationPages/llh.out"
    all_stations = pd.read_csv(url_stations, delim_whitespace=True, skiprows=1, header=None,
                           names=['Station','Latitude', 'Longitude', 'Height', 'DOMES', 'First_Year'])

    ### Stations of interest ###  
    stations_selected = []
    stations_coordinates = []
    
    for i, row in all_stations.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        if lat_south < lat < lat_north and lon_west < lon < lon_east:
            stations_selected.append(row['Station'])
            stations_coordinates.append([row['Station'], lat, lon])

    stations_coordinates_DEF = pd.DataFrame(stations_coordinates, columns=['Station', 'Latitude', 'Longitude'])
    path_stations_coordinates = os.path.join(saving_cd_coord, 'stations_coordinates.txt')
    stations_coordinates_DEF.to_csv(path_stations_coordinates, index=False)
    
    Ex_STA=[]
    #### Download stations ####
    for station in stations_selected:
        file_name = os.path.join(saving_cd, f"{station}.txt")
        link = f"http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3"
        
        ### If the link exists otherwise skip ###
        try:
            response = requests.get(link)
            print(link)
            with open(file_name, 'wb') as f:
                f.write(response.content)
            Ex_STA.append(station)
        except:
            print('Link does not exist or cannot be reached.')
      
    # Filter stations that are also present in the percorso
    filtered_stations_coordinates = stations_coordinates_DEF[stations_coordinates_DEF['Station'].isin(Ex_STA)]

    # Overwrite the stations_coordinates file with the filtered data
    filtered_stations_coordinates.to_csv((saving_cd_coord+'stations_coordinates.txt', index=False)

    return print('all done!')

saving_cd_coord = "/Users/giacomo/Documents/PhD/Papers/GNSS_DENOISER/Chile"
############ COORDINATES ############
lat_south=-55
lat_north=-5
lon_west=-81
lon_east= -66
import_stations(saving_cd_coord,lat_south,lat_north,lon_west,lon_east)
