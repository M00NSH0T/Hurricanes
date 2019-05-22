# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:26:41 2018

this downloads all the composite water vapor satellite images from the gibbs site between a 
start and end date.

example:
https://www.ncdc.noaa.gov/gibbs/image/GRD-1/WV/2004-01-01-00

@author: kylea
"""



from datetime import date, datetime, timedelta
import os
import urllib.request

#source:https://stackoverflow.com/questions/10688006/generate-a-list-of-datetimes-between-an-interval
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


url_start='https://www.ncdc.noaa.gov/gibbs/image/GRD-1/WV/'
save_dir='F:\\satellite_imagery'

for result in perdelta(datetime(2006, 1, 1), datetime(2018, 12, 13), timedelta(hours=3)):
     image_name=result.strftime("%Y-%m-%d-%H")
     print(image_name)
     full_url=url_start+image_name
     save_path=os.path.join(save_dir,image_name)
     try:
         urllib.request.urlretrieve(full_url, save_path)
     except:
         print('error: ',image_name)
