# -*- coding: utf-8 -*-，
import os
os.chdir(r'...')


import numpy as np
import datetime
import re


import multiprocessing  
import wind_power_function_2025
def multi_process(prim):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map( wind_power_function_2025.main, prim)
    out = list(results)
    return out


def find_filePath(var,climate_model,scenario,year):
    #
    #
    #    
    folder = './climate_data'

    path = 0
    try:
        file_path = os.path.join(folder, var, climate_model, scenario)
        files = os.listdir(file_path)
        for f in files:
            if int(re.split('_',f)[6][:4]) == year and re.split('_',f)[3] == scenario:
                path = os.path.join(file_path, f)
    except:
        file_path = os.path.join(folder, var)
        files = os.listdir(file_path)
        for f in files:
            if int(re.split('_',f)[6][:4]) == year and re.split('_',f)[3] == scenario and re.split('_',f)[2] == climate_model:
                path = os.path.join(file_path, f)        
    return path



import xarray as xr
scenario = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
# climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
#                   'CanESM5','CMCC-ESM2','EC-Earth3',
#                   'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
#                   'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6',
#                   #'CNRM-ESM2-1','GISS-E2-1-G', 
#                   'MPI-ESM1-2-HR', 'UKESM1-0-LL',
#                   'MRI-ESM2-0', 'NorESM2-MM']
climate_models = 'ACCESS-ESM1-5' 

for s in [0,1,2,3]:
    if s == 0:
        ys = np.arange(1985,2015)
    else:
        ys = np.arange(2015,2101)
        
    for year in ys:    
        save_name = r'./daily_power_production/wind/land/global_wind_power_' + scenario[s] + '_' + str(year) + '_' + climate_models +'.nc'
        if os.path.exists(save_name):
            continue
        #climate data
        tas = xr.open_dataset( find_filePath('tas', climate_models, scenario[s], year) )['tas']         
        tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180
        tas = tas.sortby(tas.lon) 
        tas = tas.sortby(tas.lat, ascending = False) 

        wind_speed = xr.open_dataset( find_filePath('sfcWind', climate_models, scenario[s], year) )['sfcWind'] 
        wind_speed.coords['lon'] = (wind_speed.coords['lon'] + 180) % 360 - 180
        wind_speed = wind_speed.sortby(wind_speed.lon) 
        wind_speed = wind_speed.sortby(wind_speed.lat, ascending = False) 


        # Extract length info from one file
        days = wind_speed.shape[0]
        size = wind_speed.shape[1] * wind_speed.shape[2]


        # DEM
        dem_map = xr.open_dataset('./sensitivity/data/dem.tif')['band_data'][0]
        dem_map.coords['x'] = (dem_map.coords['x'] + 180) % 360 - 180
        dem_map = dem_map.sortby(dem_map.x) 
        dem_map = dem_map.sel(y = slice(89.75, -60))
        dem_map = dem_map.sortby(dem_map.y, ascending = False) 

        #
        # read power law exponent 
        alpha_map = xr.open_dataset('./sensitivity/output/alpha/alpha_mean_season_1985-2014.nc')['alpha']
        alpha_map = alpha_map.isel(lat=alpha_map.lat > -60)
        # expand to daily and hourly  
        alpha_map = alpha_map.sel(season = wind_speed["time"].dt.season)

        pixel_type = np.full((len(wind_speed.lat), len(wind_speed.lon)), 0, dtype = np.int8)
        pixel_type[~np.isnan(wind_speed[0,:])] = 1

        # combine
        input_para = np.array(list(zip(pixel_type.reshape(-1).T, 
                                       tas.values.reshape(tas.shape[0],-1).T, 
                                       dem_map.values.reshape(-1).T, 
                                       wind_speed.values.reshape(wind_speed.shape[0],-1).T, 
                                       alpha_map.values.reshape(alpha_map.shape[0],-1).T)), 
                              dtype = object)
        print(year, 'finished: combine data', datetime.datetime.now())


        t = datetime.datetime.now()
        r = multi_process(input_para)
        print(datetime.datetime.now() - t)

        maps = np.array(r, dtype = np.float32).reshape(wind_speed.shape[1], wind_speed.shape[2], days)
        r = None

        lon = wind_speed.lon.values
        lat = wind_speed.lat.values
        time = wind_speed.time.values
        maps = xr.DataArray(maps, dims = ['lat','lon','time'], name = 'wind_power_production',
                            coords={'lat': lat,'lon': lon, 'time':time})

        maps.attrs['units'] = 'W'      
        maps.to_netcdf(r'./daily_power_production/wind/land/global_wind_power_' + scenario[s] + '_' + str(year) + '_' + climate_models +'.nc', 
                       encoding={'wind_power_production': {'zlib': True, 'complevel': 6}}) 
        del maps
        print('finished output', climate_models, scenario[s], year)

    
    
