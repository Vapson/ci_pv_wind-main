# -*- coding: utf-8 -*-
import os
os.chdir(r'...')


import pandas as pd
import numpy as np
import datetime
import re
import gc


import multiprocessing  
import pv_power_function_daily
def multi_process(prim):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map( pv_power_function_daily.main, prim)
    out = list(results)
    return out


#  multi-model mean and 10th - 90th
import xarray as xr
import rioxarray as rxr
             


year = 2010    
#
# Read the variables
#
time = xr.open_mfdataset( './data/daily/2m_temperature_daily_mean_2010*.nc', 
                         combine="by_coords", decode_times=True )['time']

tas = xr.open_dataset( './data/hourly/2m_temperature_hourly.nc')['t2m'].chunk({"valid_time": -1})
tas = tas.isel(latitude = tas.latitude > -60)
tas = tas.assign_coords(doy = tas["valid_time"].dt.dayofyear)

tasmin = tas.groupby("doy").min().compute()
tasmin = tasmin.sel(doy=time.dt.dayofyear)


tasmax = tas.groupby("doy").max().compute()
tasmax = tasmax.sel(doy=time.dt.dayofyear)


tas = tas.groupby("doy").mean().compute()
tas = tas.sel(doy=time.dt.dayofyear)


    
wind_speed = xr.open_dataset( './data/hourly/wind_speed_hourly.nc')
wind_speed = wind_speed.isel(latitude = wind_speed.latitude > -60)
wind_speed = np.sqrt(wind_speed['u10']**2 + wind_speed['v10']**2)
wind_speed = wind_speed.assign_coords(doy = wind_speed["valid_time"].dt.dayofyear)
wind_speed = wind_speed.chunk({"valid_time": -1}).groupby("doy").mean().compute()
wind_speed = wind_speed.sel(doy=time.dt.dayofyear)


rsds = xr.open_dataset( './data/hourly/Surface_solar_radiation_downwards_hourly.nc')['ssrd'].chunk({"valid_time": -1})
rsds = rsds.isel(latitude = rsds.latitude > -60)

rsds['valid_time'] = rsds['valid_time'] - pd.Timedelta(hours=1)
rsds = rsds.assign_coords(doy = rsds["valid_time"].dt.dayofyear)    
rsds['valid_time'].values[rsds['valid_time'].dt.year == year-1] = rsds['valid_time'][rsds['valid_time'].dt.year == year-1] + pd.Timedelta(days=365)
rsds = rsds.sortby('valid_time')
rsds = rsds.chunk({"valid_time": -1}).groupby("doy").sum().compute()
rsds = rsds.sel(doy=time.dt.dayofyear)






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
#time = wind_speed.valid_time.values
years = np.array([year for i in range(size)])



# Extract length info from one file
days = wind_speed.shape[0]
size = wind_speed.shape[1] * wind_speed.shape[2]

# Extract lon/lat info from one file
lat = wind_speed.latitude.values
lon = wind_speed.longitude.values
lons, lats = np.meshgrid(lon,lat) 
#time = wind_speed.time.values


# decom_method
method = 'Pramod' # /'Hourly_Ex'/'liu_jordan'
decom_method = np.array([method for i in range(size)])


# combine
# year, rs, temp_mean, temp_min, temp_max, wind_speed, lon, lat, aroof, sfroof, gcr
input_para = np.array(list(zip(pixel_type.reshape(-1).T, 
                               years, 
                               rsds.values.reshape(rsds.shape[0],-1).T / 86400, 
                               wind_speed.values.reshape(wind_speed.shape[0],-1).T, 
                               tas.values.reshape(tas.shape[0],-1).T, 
                               tasmin.values.reshape(tasmin.shape[0],-1).T, 
                               tasmax.values.reshape(tasmax.shape[0],-1).T, 
                               lons.reshape(-1).T, 
                               lats.reshape(-1).T,
                               decom_method,                                
                           )), dtype=object)

del tasmax, tasmin, tas, wind_speed, rsds
gc.collect()
print(year, 'finished: combine data',datetime.datetime.now())


t = datetime.datetime.now()
r = multi_process(input_para)
print(datetime.datetime.now() - t )
input_para = None

maps = np.array(r, dtype = np.float32).reshape(len(lat), len(lon), len(time))   
r = None

maps = xr.DataArray(maps, dims = ['lat','lon','time'], name = 'pv_power_production',
                    coords={'lat': lat,'lon': lon, 'time':time.values})

maps.attrs['units'] = 'Wh'       
maps.to_netcdf(r'./output/global_pv_power_' + str(year) + '_daily_' + method + '.nc', 
               encoding={'pv_power_production': {'zlib': True, 'complevel': 6}}) 
del maps
print('finished output',year)




