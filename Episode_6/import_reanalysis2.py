# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:18:19 2019

@author: kylea
"""
#import h5netcdf.legacyapi as netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import os
from multiprocessing import Pool
import sys
import time



#for year in years:
#year=1979
def load_year(year,level_opt='partial'):
    conn = create_engine('mysql://moonshot:episode3@localhost:3306/weather')
    variables=['air','hgt','omega','rhum','uwnd','vwnd']
    #level_indices=[16,11,5,0]
    directory='T:\Reanalysis2\pressure'
    arrays=[]
    for variable in variables:
        file=f"{variable}.{year}.nc"      
        full_path=os.path.join(directory,file)  
        print(full_path)
        
        netcdf_data = Dataset(full_path, mode='r')        
        
        lons = netcdf_data.variables['lon'][:]        
        lats = netcdf_data.variables['lat'][:]
        time=netcdf_data.variables['time'][:]
        if level_opt=='all':
            levels=netcdf_data.variables['level'][:]
        else:
            levels=[10,100,500,1000]#from gdas... replaced 0 with 10, but might add surface later

        values = netcdf_data.variables[variable][:]
        arrays.append(values)
        netcdf_data.close()
    
    #all_data=np.stack(arrays)  
    dates = [datetime(year,1,1)+n*timedelta(hours=6) for n in range(len(time))]
    for dt_index in range(len(time)):
        for l in range(len(levels)):
            output=pd.DataFrame()
            for v in range(len(arrays)):
                temp=arrays[v]
                df=pd.DataFrame(np.squeeze(temp[dt_index,l,:,:]),index=lats,columns=lons)
                df=df.stack(dropna=True)
                df=pd.DataFrame(df)
                df.reset_index(inplace=True)
                df.columns=['lat','lon',variables[v]]
        #            df.columns=[variable]
#                df['level']=levels[l]
                df['datetime']=dates[dt_index]
                df.set_index(['datetime','lat','lon'],inplace=True)
                output[variables[v]]=df[df.columns[0]].copy()
    #        output[variable]=var_df
            output_file="reanalysis2_level_"+str(levels[l])

            output.to_sql(output_file,conn,if_exists='append',index=True)
            print("dt: ",dates[dt_index])

       
if __name__=='__main__':
        #inserted due to https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    years=range(1979,2020)


    f1=years[:int(len(years)/4)]
    f2=years[int(len(years)/4):int(2*len(years)/4)]
    f3=years[int(2*len(years)/4):int(3*len(years)/4)]
    f4=years[int(3*len(years)/4):]
##    read_gdas(files[4])
##    dates=monthlist_fast(["2005-03-01","2018-11-01"])
    
#    files.rev.erse()

    for y in years:
        load_year(y)



#    time1=time.time()
#    pool=Pool(processes=3)
#    output=pool.map(load_year,years,1)
#    pool.close()
#    pool.join()  
#    total_time=time.time()-time1    
#    print("process took: ",total_time/60," minutes")
#    for year in years:
#        load_year(year)
              
            




