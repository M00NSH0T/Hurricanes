# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:18:19 2019

@author: kylea
"""
#import h5netcdf.legacyapi as netCDF4
from netCDF4 import Dataset
import numpy as np
import os
import pickle


#for year in years:
#year=1979
def load_year(year):
    variables=['air','hgt','omega','rhum','uwnd','vwnd']
    #level_indices=[16,11,5,0]
    directory='T:\Reanalysis2\pressure'
    arrays=[]
    for variable in variables:
        file=f"{variable}.{year}.nc"      
        full_path=os.path.join(directory,file)  
        print(full_path)        
        netcdf_data = Dataset(full_path, mode='r')        
        values = netcdf_data.variables[variable][:]
        arrays.append(values)
        netcdf_data.close()
    output_dir="T:\Reanalysis2\pkl_files"
    all_data=np.stack(arrays)  
    output_file=f"reanalysis2_{year}.pkl"
    full_output_path=os.path.join(output_dir,output_file)  
    
#    apparently there's a bug in Python3 that prevents np.save or pickle.dump from saving onjects >3GB
#    https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
#    workaround from stackoverflow
    with open(full_output_path, 'wb') as pickle_file:
        pickle.dump(all_data,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)
#    np.save(full_output_path,all_data,fix_imports=False)
#    print("year: ",year)

       
if __name__=='__main__':
        #inserted due to https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    years=range(1979,2020)


    for y in years:
        load_year(y)

#    load_year(1980)



            




