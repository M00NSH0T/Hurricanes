# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:35:32 2019

This is a modified version of the utility provided in pix2pix that combines
pairs of input and output training images together. 

primary source (unless credited in comment above code chunk):
https://github.com/phillipi/pix2pix/blob/master/scripts/combine_A_and_B.py
"""

import os
import shutil
from pdb import set_trace as st
import os
import numpy as np
import cv2
#from sklearn import train_test_split
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta

#you'll need to setup your own connection. 
conn = create_engine('mysql://your_username:your_password@localhost:3306/weather')


#source: http://techs.studyhorror.com/python-copy-rename-files-i-122
def copy_rename(old_file_name, new_file_name):
        src_dir= os.curdir
        dst_dir= os.path.join(os.curdir , "subfolder")
        src_file = os.path.join(src_dir, old_file_name)
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        os.rename(dst_file, new_dst_file_name)


#code will run if the directories don't exist with no error, but
#files will never actually get saved. Must create empty directories
def create_directories(output_root):
    try:
        os.mkdir(output_root)
        print("Directory " , output_root ,  " Created ") 
    except FileExistsError:
        print("Directory " , output_root ,  " already exists") 
    for sub_dir in ('train','val','test'):
        output_dir=os.path.join(output_root,sub_dir)
        try:
            os.mkdir(output_dir)
            print("Directory " , output_dir ,  " Created ") 
        except FileExistsError:
            print("Directory " , output_dir ,  " already exists")  

splits=['train','test','val']

query="select distinct Date_Time from track_data where Date_Time>'2004'"
dts=pd.read_sql(query,conn)

#the shift function lines up out forecast. In this case, it's just looking one timestep (6 hours),
#but eventually, we'll want this to look out farther... 48 hours or so. This was just a quick proof of concept.
dts['future']=dts['Date_Time']+timedelta(hours=48)#change to 48 for real run


dts.dropna(inplace=True)
train, validate, test = np.split(dts.sample(frac=1), [int(.6*len(dts)), int(.8*len(dts))])
train.reset_index(inplace=True,drop=True)
train.reset_index(inplace=True)
test.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True)
validate.reset_index(inplace=True,drop=True)
validate.reset_index(inplace=True)
train['set']='train'
validate['set']='val'
test['set']='test'
dts=train.append([test,validate])


#change to location where you stored your original satellite images (or whatever you want to use)
#just be aware, the names of the images must be in the satellite image format with date and hour in the name
img_dir='F:\\satellite_imagery'

#this is where the combined A_B concatenated image pairs will be stored.
output_root='M:\\satellite_imagery'
create_directories(output_root)

for index,row in dts.iterrows():
        name_B = row['future'].strftime("%Y-%m-%d-%H")
        print(index," ",name_B)
        name_A = row['Date_Time'].strftime("%Y-%m-%d-%H")
        
        path_A = os.path.join(img_dir, name_A)
        path_B = os.path.join(img_dir, name_B)        
        output_dir=os.path.join(output_root,row['set'])
        
  
                
        name_AB=str(row['index']+1)+".jpg"
        if os.path.isfile(path_A) and os.path.isfile(path_B):

            path_AB = os.path.join(output_dir, name_AB)
            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
