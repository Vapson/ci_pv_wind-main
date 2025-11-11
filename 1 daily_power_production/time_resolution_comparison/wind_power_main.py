# -*- coding: utf-8 -*-
import os
os.chdir(r'...')


import numpy as np
import datetime
import re
import gc

#multi_process
import multiprocessing  
import wind_power_function
def multi_process(prim):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map( wind_power_function.main, prim)
    out = list(results)
    return out


import xarray as xr
import rioxarray as rxr
year = 2010 
daily = True
hourly = False

#
#### read climate data:
if hourly:
    tas = xr.open_dataset( './data/hourly/2m_temperature_hourly.nc')['t2m']
    tas = tas.isel(latitude = tas.latitude > -60)
    
    wind_speed = xr.open_dataset( './data/hourly/wind_speed_hourly.nc')
    wind_speed = wind_speed.isel(latitude = wind_speed.latitude > -60)
    wind_speed = np.sqrt(wind_speed['u10']**2 + wind_speed['v10']**2)
    #wind_speed = wind_speed.assign_coords(doy = wind_speed["valid_time"].dt.dayofyear)
    #wind_speed = wind_speed.assign_coords(date = wind_speed["valid_time"].dt.date)
    

if daily:
    tas = xr.open_mfdataset( './data/daily/2m_temperature_daily_mean_2010*.nc', combine="by_coords", decode_times=True )['t2m']   
    # tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180
    tas = tas.sortby(tas.lon) 
    tas = tas.sortby(tas.lat, ascending = False) 
    tas = tas.isel(lat=tas.lat > -60)

    #  DOY --> daily average wind speed data
    wind_speed = xr.open_mfdataset( './data/hourly/wind_speed_hourly.nc')
    wind_speed = wind_speed.isel(latitude = wind_speed.latitude > -60)
    wind_speed = np.sqrt(wind_speed['u10']**2 + wind_speed['v10']**2)
    wind_speed = wind_speed.assign_coords(doy = wind_speed["valid_time"].dt.dayofyear)
    wind_speed = wind_speed.groupby("doy").mean()
    wind_speed = wind_speed.sel(doy=tas["time"].dt.dayofyear)

    
    
    
#
#### read pixel type:
land = rxr.open_rasterio('./data/land.tif')[0,:].values
dis2coastline = rxr.open_rasterio('./data/dis2coastline.tif')[0,:].values

# creat mask 
onshore = land == 1
offshore = (land==0) & (dis2coastline<200*1000) & (dis2coastline!=-1)
pixel_type = np.full((len(wind_speed.latitude), len(wind_speed.longitude)), 0, dtype = np.int8)
pixel_type[onshore] = 1
pixel_type[offshore] = 2


#
#### read dem:
dem_map = xr.open_dataset('./data/dem.tif')['band_data'][0,:]
dem_map.coords['x'] = (dem_map.coords['x'] + 180) % 360 - 180
dem_map = dem_map.sortby(dem_map.x) 
dem_map = dem_map.isel(y=dem_map.y > -60)

#
# read power law exponent 
alpha_map = xr.open_dataset('./output/alpha/alpha_mean_season_1985-2014.nc')['alpha']
alpha_map = alpha_map.isel(lat=alpha_map.lat > -60)
# expand to daily and hourly
if hourly:
    alpha_map = alpha_map.sel(season = wind_speed["valid_time"].dt.season)
    time = wind_speed.valid_time.values
if daily:      
    alpha_map = alpha_map.sel(season = wind_speed["time"].dt.season)
    time = wind_speed.time.values



# Extract length info from one file
days = wind_speed.shape[0]
size = wind_speed.shape[1] * wind_speed.shape[2]

# Extract lon/lat info from one file
lat = wind_speed.latitude.values
lon = wind_speed.longitude.values
lons, lats = np.meshgrid(lon,lat) 



# combine and convert to list
# pixel_type, Tas, dem, wind_speed, hellman_exponent
input_para = np.array(list(zip(pixel_type.reshape(-1).T, 
                               tas.values.reshape(tas.shape[0],-1).T, 
                               dem_map.values.reshape(-1).T, 
                               wind_speed.values.reshape(wind_speed.shape[0],-1).T, 
                               alpha_map.values.reshape(alpha_map.shape[0],-1).T)), dtype=object)
del tas, wind_speed, dem_map, alpha_map
gc.collect()
print(year, 'finished: combine data',datetime.datetime.now())



print(datetime.datetime.now())
r = multi_process(input_para)
print(datetime.datetime.now())
input_para = None

maps = np.array(r, dtype = np.float32).reshape(len(lat), len(lon), len(time))   
r = None
maps = xr.DataArray(maps, dims = ['lat', 'lon','time'], name = 'wind_power_production',
                    coords={'lat': lat, 'lon': lon, 'time':time})

maps.attrs['units'] = 'W'       
if hourly:
    maps.to_netcdf(r'./output/global_wind_power_' + str(year) +'_hourly.nc', 
                   encoding={'wind_power_production': {'zlib': True, 'complevel': 6}}) 
    
if daily:
    maps.to_netcdf(r'./output/global_wind_power_' + str(year) +'_daily.nc', 
                   encoding={'wind_power_production': {'zlib': True, 'complevel': 6}}) 
print('finished output', year)  



