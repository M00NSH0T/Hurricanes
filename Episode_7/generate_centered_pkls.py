# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:15:49 2019

@author: kylea
"""

from sqlalchemy import create_engine
import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime,timedelta

conn = create_engine('mysql://moonshot:3p1s0d37@localhost:3306/weather')

# https://stackoverflow.com/questions/4265546/python-round-to-nearest-05
# function to round to nearest 2.5 degrees, the resolution of Reanalysis2 data
#def round_degrees(number):
#    return (round(number / 2.5) * 2.5)

track_data=pd.read_sql("SELECT Date_Time,Adj_Lat,Adj_Lon FROM weather.track_data",conn)
#year=1980

def center_year(year):
    print(year)
    data_dir="T:\Reanalysis2\pkl_files"

    #create two numpy arrays containing the lat/lon that correspond to the rows/columns of our numpy arrays
    #lats=np.arange(90,-90.1,-2.5)
    #lons=np.arange(-180,180.1,2.5)

    file=f"reanalysis2_{year}.pkl"
    full_path=os.path.join(data_dir,file)  
    with open(full_path, 'rb') as pickle_file:
        fulldataset=pickle.load(pickle_file)  

    #track_data['Lat_Round']=track_data.loc[:,'Adj_Lat'].apply(round_degrees)
    #track_data['Lon_Round']=track_data.loc[:,'Adj_Lon'].apply(round_degrees)

    dts=[datetime(year,1,1)+n*timedelta(hours=6) for n in range(fulldataset.shape[1])]

    track_1yr=track_data[track_data.loc[:,'Date_Time'].dt.year==year]


    outputs=[]
    for i,row in track_1yr.iterrows():
        try:
            index=dts.index(row['Date_Time'])
            data_cube=fulldataset[:,index,:,:,:].filled(0)#convert from masked array to np array with 0 replacing missing values (shouldn't be any, but just in case...)
            Lat=row['Adj_Lat']
            Lat_Shift=round(Lat*data_cube.shape[2]/180)#center is zero lat
            Lon=row['Adj_Lon']
            Lon_Shift=round(-Lon*data_cube.shape[3]/360)#center is also zero, but roll goes wrong way
            data_cube=np.roll(data_cube,(Lat_Shift,Lon_Shift),(2,3))
            outputs.append(data_cube)
        except:
            print("problem with: ", row['Date_Time']) #this gets triggered when we encounter a time in the track data that isn't in the set [00:00:00,06:00:00,12:00:00,18:00:00]

    outputs=np.stack(outputs,axis=1)

    #output_dir="T:\Reanalysis2\pkl_files_centered"
    output_file=f"centered_{year}.pkl"
    full_output_path=os.path.join(data_dir,output_file)

    with open(full_output_path, 'wb') as pickle_file:
        pickle.dump(outputs,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
        #inserted due to https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    years=range(1983,2018)


    for y in years:
        center_year(y)

#    load_year(1980)
