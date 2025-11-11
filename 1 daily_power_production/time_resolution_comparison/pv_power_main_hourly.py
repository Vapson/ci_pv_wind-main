# -*- coding: utf-8 -*-
import os
os.chdir(r'...')

import pandas as pd
import numpy as np
import datetime
import gc

# multi_process
import multiprocessing  
import pv_power_function_hourly
def multi_process(prim):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map( pv_power_function_hourly.main, prim)
    out = list(results)
    return out


import xarray as xr
import rioxarray as rxr
year = 2010 

tas = xr.open_dataset( './data/hourly/2m_temperature_hourly.nc')['t2m']
tas = tas.isel(latitude = tas.latitude > -60)
    
wind_speed = xr.open_dataset( './data/hourly/wind_speed_hourly.nc')
wind_speed = wind_speed.isel(latitude = wind_speed.latitude > -60)
wind_speed = np.sqrt(wind_speed['u10']**2 + wind_speed['v10']**2)

rsds = xr.open_dataset( './data/hourly/Surface_solar_radiation_downwards_hourly.nc')['ssrd']
rsds = rsds.isel(latitude = rsds.latitude > -60)
# The timestamp for the cumulative amount in ERA5 is defined as the end time.
rsds['valid_time'] = rsds['valid_time'] - pd.Timedelta(hours=1)  
rsds['valid_time'].values[rsds['valid_time'].dt.year == year-1] = rsds['valid_time'][rsds['valid_time'].dt.year == year-1] + pd.Timedelta(days=365)
rsds = rsds.sortby('valid_time')


#
#### read pixel type:
land = rxr.open_rasterio('./data/land.tif')[0,:].values
# creat mask 
onshore = land == 1
pixel_type = np.full((len(wind_speed.latitude), len(wind_speed.longitude)), 0, dtype = np.int8)
pixel_type[onshore] = 1


# Extract length info from one file
days = wind_speed.shape[0]
size = wind_speed.shape[1] * wind_speed.shape[2]

# Extract lon/lat info from one file
lat = wind_speed.latitude.values
lon = wind_speed.longitude.values
lons, lats = np.meshgrid(lon,lat) 
time = wind_speed.valid_time.values
years = np.array([year for i in range(size)])
times = np.array([time for i in range(size)])




# combine
input_para = np.array(list(zip(pixel_type.reshape(-1).T, 
                               years, 
                               times,
                               rsds.values.reshape(rsds.shape[0],-1).T, 
                               wind_speed.values.reshape(wind_speed.shape[0],-1).T, 
                               tas.values.reshape(tas.shape[0],-1).T, 
                               lons.reshape(-1).T, 
                               lats.reshape(-1).T,
                               
                           )), dtype=object)
del tas, wind_speed, rsds, times
gc.collect()
print(year, 'finished: combine data',datetime.datetime.now())


     
t = datetime.datetime.now()
r = multi_process(input_para)
print(datetime.datetime.now() - t)
input_para = None

maps = np.array(r, dtype = np.float32).reshape(len(lat), len(lon), len(time))   
r = None

maps = xr.DataArray(maps, dims = ['lat','lon','time'], name = 'pv_power_production',
                    coords={'lat': lat,'lon': lon, 'time':time})

maps.attrs['units'] = 'W'       
maps.to_netcdf(r'./output/global_pv_power_' + str(year) + '_hourly_r4.nc', 
               encoding={'pv_power_production': {'zlib': True, 'complevel': 6}}) 
print('finished output', year)

maps = xr.open_dataset(r'./output/global_pv_power_' + str(year) + '_hourly_r4.nc').chunk({"time": -1})
maps = maps.assign_coords(doy = maps["time"].dt.dayofyear)
maps = maps.groupby("doy").sum().compute()
maps.attrs['units'] = 'Wh'  
maps.to_netcdf(r'./output/global_pv_power_2010_hourly_agg_to_daily_r4.nc', 
                encoding={'pv_power_production': {'zlib': True, 'complevel': 6}})
print('finished output aggdaily', year)

