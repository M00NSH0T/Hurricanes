# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:45:37 2019
this imports the track data found in "hurricane_data_raw.csv" which is just the hurdate2 data you
can pull from the NHC website.

@author: kylea
"""

import pandas as pd
from sqlalchemy import create_engine

conn = create_engine('mysql://your_username:your_password@localhost:3306/weather')

track_data=pd.read_csv('hurricane_data_raw.csv',parse_dates=['Date_Time','Date','Time'],infer_datetime_format=True)
track_data=track_data[['Hurricane_ID','Date_Time','unknown1', 'Status', 'Lat', 'Lon',
       'Max_Speed', 'Min_Pressure', 'NE_34', 'SE_34', 'SW_34', 'NW_34',
       'NE_50', 'SE_50', 'SW_50', 'NW_50', 'NE_64', 'SE_64', 'SW_64', 'NW_64',
       'Adj_Lat', 'Adj_Lon']]

track_data.to_sql('track_data',index=False,con=conn)