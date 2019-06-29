# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 06:43:11 2019

@author: kylea
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from datetime import timedelta
from joblib import dump

#make sure to update your credentials (and make something secure)
conn = create_engine('mysql://username:password@localhost:3306/weather')

data_dir="T:\Reanalysis2\pkl_files"
file_out="encoded_data.pkl"
full_path=os.path.join(data_dir,file_out)
grid_data=pd.read_pickle(full_path)

track_data=pd.read_sql('track_data',conn)
#track_data.set_index(['Hurricane_ID','Date_Time'],inplace=True)

#setup training data
Present=track_data[['Hurricane_ID','Date_Time','Adj_Lat','Adj_Lon']].copy()




#we're going to want to know where the storm will be in 48 hours for training
Future=Present[['Date_Time','Hurricane_ID']].copy()
Future['Forecast_Time']=Present['Date_Time']+timedelta(hours=48)

#and we're going to want to know where's it just came from so we can a sense for it's direction and speed
Past=Present[['Date_Time','Hurricane_ID']].copy()
Past['Last_Timestep']=Present['Date_Time']-timedelta(hours=6)


Future=Future.merge(Present,how='inner',left_on=['Hurricane_ID','Forecast_Time'],right_on=['Hurricane_ID','Date_Time'],suffixes=("","_right"))
Past=Past.merge(Present,how='inner',left_on=['Hurricane_ID','Last_Timestep'],right_on=['Hurricane_ID','Date_Time'],suffixes=("","_right"))
Future=Future[['Hurricane_ID','Date_Time','Adj_Lat','Adj_Lon']].copy()
Past=Past[['Hurricane_ID','Date_Time','Adj_Lat','Adj_Lon']].copy()


Present=Present.merge(Future,how='inner',left_on=['Hurricane_ID','Date_Time'],right_on=['Hurricane_ID','Date_Time'],suffixes=("","_Future"))
Present=Present.merge(Past,how='inner',left_on=['Hurricane_ID','Date_Time'],right_on=['Hurricane_ID','Date_Time'],suffixes=("","_Past"))

#add some time of day / year info to the input and scale everything from 0 to 1
Present['day_of_year']=Present['Date_Time'].dt.dayofyear/365
Present['hour']=Present['Date_Time'].dt.hour/24
Present['Adj_Lat']=(Present['Adj_Lat']+90)/180
Present['Adj_Lon']=(Present['Adj_Lon']+180)/360
Present['Adj_Lat_Past']=(Present['Adj_Lat_Past']+90)/180
Present['Adj_Lon_Past']=(Present['Adj_Lon_Past']+180)/360

#add in the grid data
Present=Present.merge(grid_data,how='left',left_on='Date_Time',right_index=True)
Present.dropna(inplace=True)


train=Present[(Present['Date_Time']>'1978') & (Present['Date_Time']<'2013')]
test=Present[Present['Date_Time']>='2013']
train.set_index(['Date_Time','Hurricane_ID'],inplace=True)
test.set_index(['Date_Time','Hurricane_ID'],inplace=True)
#X_train=train[['day_of_year','hour', 'Adj_Lat', 'Adj_Lon','Adj_Lat_Past', 'Adj_Lon_Past']]
X_train=train.drop(columns=['Adj_Lat_Future','Adj_Lon_Future'])
y_train=train[['Adj_Lat_Future','Adj_Lon_Future']]

X_test=test.drop(columns=['Adj_Lat_Future','Adj_Lon_Future'])
y_test=test[['Adj_Lat_Future','Adj_Lon_Future']]

#setup y (output for training set)
#output=

print('Starting Forest Training')
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=1000,verbose=2)#725 minutes per 1000 trees
regr.fit(X_train, y_train)  

print('saving trained forest model')
forest_file='forest_trained.joblib'
full_forest_path=full_path=os.path.join(data_dir,forest_file)

dump(regr, full_forest_path)
