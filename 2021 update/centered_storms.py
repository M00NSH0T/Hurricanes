import requests
import netCDF4
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import gc
import os
from math import pi
os.environ['PROJ_LIB'] = r'c:\Users\kylea\AppData\Local\conda\conda\envs\hurricanes\Library\share'

from mpl_toolkits.basemap import Basemap
from pyproj import Proj
import numpy as np


def get_s3_keys(bucket, s3_client, prefix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """

    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def gen_image(m,year=2017, day_of_year=60, hour=0, band=12, margin=10, center_lat=45, center_lon=-70, SID='', my_dpi=200,
               save_img=True, show_lines=True):
    gc.collect()
    try:
        keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M3C{band:02.0f}'
                      )
        all_keys= [key for key in keys]
        key = all_keys[0]
    except:
        keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}'
                      )
        all_keys= [key for key in keys]
        key = all_keys[0]
    resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')

    file_name = key.split('/')[-1].split('.')[0]
    g16nc = netCDF4.Dataset(file_name, memory=resp.content)

    height = g16nc.variables['goes_imager_projection'].perspective_point_height

    # Longitude of the satellite's orbit
    lon = g16nc.variables['goes_imager_projection'].longitude_of_projection_origin
    lat = g16nc.variables['goes_imager_projection'].latitude_of_projection_origin

    # Which direction do we sweep in?
    sweep = g16nc.variables['goes_imager_projection'].sweep_angle_axis

    X = g16nc.variables['x'][:] * height
    Y = g16nc.variables['y'][:] * height
    rad = g16nc.variables['Rad'][:]

    fig = plt.figure(figsize=(300 / my_dpi, 300 / my_dpi), dpi=my_dpi)
    #     fig = plt.figure(figsize=(6,6),dpi=200)
    # m = Basemap(resolution='l', projection='geos', lon_0=lon, lat_0=lat, rsphere=(6378137.00, 6356752.3142))
    # #             llcrnrx=X.min(), llcrnry=Y.min(), urcrnrx=X.max(), urcrnry=Y.max())
    #     m.fillcontinents(color='coral',lake_color='aqua')

    if show_lines:
        #         m.drawlsmask()
        m.bluemarble()
    #         m.drawcountries()
    #         m.drawcoastlines()
    #         m.drawstates()
    m.imshow(np.flipud(rad).astype('uint8'), cmap='gray', vmin=0, vmax=100, alpha=0.6)  # make it black / white

    lllon = center_lon - margin
    urlon = center_lon + margin
    lllat = center_lat - margin
    urlat = center_lat + margin

    xmin, ymin = m(lllon, lllat)
    xmax, ymax = m(urlon, urlat)
    xrange = xmax - xmin
    yrange = ymax - ymin
    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    usable_range = min(xrange, yrange)
    xmin = xcenter - usable_range / 2
    xmax = xcenter + usable_range / 2
    ymin = ycenter - usable_range / 2
    ymax = ycenter + usable_range / 2
    ax = plt.gca()

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if save_img:
        plt.axis('off')
        centered_filename = rf"./storm_centered/centered_{SID}_{year}{day_of_year}{hour}.png"
        plt.savefig(centered_filename, bbox_inches='tight', pad_inches=0, transparent=True);
    #         fig=plt.figure(figsize=(300/my_dpi, 300/my_dpi), dpi=my_dpi)

    #         plt.imshow(ds_zarr.Rad, cmap='gray')
    #         plt.axis('off')
    #         disk_filename=rf"./full_disks/fulldisk_{year}{day_of_year}{hour}.png"
    #         plt.savefig(disk_filename, bbox_inches='tight',pad_inches=0,transparent=True)
    else:
        plt.show()
    plt.close(fig)

if __name__=='__main__':


    track_data = pd.read_csv('clean_tracks.csv', usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR'],
                             infer_datetime_format=True)

    track_data = pd.read_csv('clean_tracks.csv', usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR'],
                             infer_datetime_format=True)
    track_data = track_data.replace(' ', np.nan)  # a few hunderd blank rows get brought in at the end.
    track_data.dropna(inplace=True)
    track_data['SID'] = track_data['SID'].astype('string')
    track_data['ISO_TIME'] = pd.to_datetime(track_data['ISO_TIME'])
    track_data['LAT'] = track_data['LAT'].astype('float16')
    track_data['LON'] = track_data['LON'].astype('float16')
    track_data['STORM_SPEED'] = track_data['STORM_SPEED'].astype('int8')
    track_data['STORM_DIR'] = track_data['STORM_DIR'].astype('int8')
    track_data['sin_STORM_DIR'] = np.sin(np.deg2rad(track_data['STORM_DIR']))
    track_data['cos_STORM_DIR'] = np.cos(np.deg2rad(track_data['STORM_DIR']))
    track_data['year'] = track_data['ISO_TIME'].dt.year

    track_data['day_of_year'] = track_data['ISO_TIME'].dt.day_of_year
    track_data['sin_day_of_year'] = np.sin(2 * pi * track_data['day_of_year'] / 364)
    track_data['cos_day_of_year'] = np.cos(2 * pi * track_data['day_of_year'] / 364)

    track_data['hour'] = track_data['ISO_TIME'].dt.hour
    track_data['sin_hour'] = np.sin(2 * pi * track_data['hour'] / 23)
    track_data['cos_hour'] = np.cos(2 * pi * track_data['hour'] / 23)
    track_data['LON'][track_data['LON'] > 180] -= 360
    track_data['LON'][track_data['LON'] < -180] += 360

    track_data.sort_values(['SID', 'ISO_TIME'], inplace=True)
    track_data.reset_index(inplace=True, drop=True)
    storm_list = track_data['SID'].drop_duplicates()
    dfs = []
    for storm in storm_list:
        temp = track_data[track_data['SID'] == storm].copy()
        temp['delta_lat'] = temp['LAT'].diff(1)
        temp['delta_lon'] = temp['LON'].diff(1)
        for offset in range(1, 5):
            temp[f'delta_lat_m{offset * 3}'] = temp['delta_lat'].shift(-offset)
            temp[f'delta_lon_m{offset * 3}'] = temp['delta_lon'].shift(-offset)
        temp.drop(columns=['delta_lat', 'delta_lon'], inplace=True)
        dfs.append(temp)
    track_data = pd.concat(dfs)

    oldest_available = datetime(year=2017, month=1, day=1) + timedelta(days=60)
    track_data = track_data[track_data['ISO_TIME'] > oldest_available]

    track_data.reset_index(inplace=True,drop=True)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    year = 2017
    day_of_year = 60
    hour = 0
    band = 14
    bucket_name = 'noaa-goes16'
    product_name = 'ABI-L1b-RadF'
    keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix=f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M3C{band:02.0f}'
                       )
    all_keys = [key for key in keys]
    key = all_keys[0]  # selecting the first measurement taken within the hour
    file_name = key.split('/')[-1].split('.')[0]

    resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    nc4_ds = netCDF4.Dataset(file_name, memory=resp.content)
    height = nc4_ds.variables['goes_imager_projection'].perspective_point_height

    # Longitude of the satellite's orbit
    lon = nc4_ds.variables['goes_imager_projection'].longitude_of_projection_origin
    lat = nc4_ds.variables['goes_imager_projection'].latitude_of_projection_origin

    X = nc4_ds.variables['x'][:] * height
    Y = nc4_ds.variables['y'][:] * height
    num_lons, num_lats  = len(X), len(Y)

    min_lon = nc4_ds.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude
    max_lon = nc4_ds.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
    min_lat = nc4_ds.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
    max_lat = nc4_ds.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    margin = 15
    track_data = track_data[track_data['LAT'] > min_lat + margin]
    track_data = track_data[track_data['LAT'] < max_lat - margin]
    track_data = track_data[track_data['LON'] > min_lon + margin]
    track_data = track_data[track_data['LON'] < max_lon - margin]
    track_data.reset_index(inplace=True, drop=True)

    m = Basemap(resolution='l', projection='geos', lon_0=lon, lat_0=lat, rsphere=(6378137.00, 6356752.3142))

    errors=[]
    for i,row in track_data.iterrows():
        print(i)
        try:
            gen_image(m,year=row['year'],day_of_year=row['day_of_year'],band=16,margin=margin,hour=row['hour'],center_lat=row['LAT'],center_lon=row['LON'],SID=row['SID'],my_dpi=300,save_img=True,show_lines=True)
        except:
            print("error: ",row)
            errors.append(row)
        gc.collect()
    error_df=pd.concat(errors,axis=0)
    error_df.to_pickle('image_gen_errors.pkl')