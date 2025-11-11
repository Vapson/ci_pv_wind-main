# -*- coding: utf-8 -*-
import os
os.chdir(r'/...')


import numpy as np
import datetime
import re
import gc

#multi_process
import multiprocessing  
import pv_power_function_2025
def multi_process(prim):
    pool =multiprocessing.Pool(multiprocessing.cpu_count())
    results=pool.map( pv_power_function_2025.main, prim)
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
            if f.endswith('.nc') and int(re.split('_',f)[6][:4]) == year and re.split('_',f)[3] == scenario and re.split('_',f)[2] == climate_model:
                path = os.path.join(file_path, f)        
    return path

            


#  multi-model mean and 10th - 90th
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
        save_name = r'./daily_power_production/pv/global_pv_power_' + scenario[s] + '_' + str(year) + '_' + climate_models +'.nc'
        if os.path.exists(save_name):
            continue
        #
        # Read the variables
        #
        tasmax = xr.open_dataset( find_filePath('tasmax', climate_models, scenario[s], year) )['tasmax']  
        tasmax.coords['lon'] = (tasmax.coords['lon'] + 180) % 360 - 180
        tasmax = tasmax.sortby(tasmax.lon) 
        tasmax = tasmax.sortby(tasmax.lat, ascending = False) 

        
        
        tasmin = xr.open_dataset( find_filePath('tasmin',climate_models,scenario[s], year) )['tasmin'] 
        tasmin.coords['lon'] = (tasmin.coords['lon'] + 180) % 360 - 180
        tasmin = tasmin.sortby(tasmin.lon) 
        tasmin = tasmin.sortby(tasmin.lat, ascending = False) 

        
        
        tas = xr.open_dataset( find_filePath('tas', climate_models, scenario[s], year) )['tas']   
        tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180
        tas = tas.sortby(tas.lon) 
        tas = tas.sortby(tas.lat, ascending = False) 
    
        
        
        rsds = xr.open_dataset( find_filePath('rsds', climate_models, scenario[s], year) )['rsds'] 
        rsds.coords['lon'] = (rsds.coords['lon'] + 180) % 360 - 180
        rsds = rsds.sortby(rsds.lon) 
        rsds = rsds.sortby(rsds.lat, ascending = False) 
        rsds.values[rsds.values<0] = 0
        rsds.values[rsds.values>600] = 600
    
        
        
        wind_speed = xr.open_dataset( find_filePath('sfcWind', climate_models, scenario[s], year) )['sfcWind'] 
        wind_speed.coords['lon'] = (wind_speed.coords['lon'] + 180) % 360 - 180
        wind_speed = wind_speed.sortby(wind_speed.lon) 
        wind_speed = wind_speed.sortby(wind_speed.lat, ascending = False)   
     
        
        
        pixel_type = np.full((len(tasmax.lat), len(tasmax.lon)), 0, dtype = np.int8)
        pixel_type[~np.isnan(tasmax[0,:])] = 1
        
        
        # Extract length info from one file
        days = wind_speed.shape[0]
        size = wind_speed.shape[1] * wind_speed.shape[2]
        
        # Extract lon/lat info from one file
        lat = wind_speed.lat.values
        lon = wind_speed.lon.values
        lons, lats = np.meshgrid(lon,lat) 
        time = wind_speed.time.values
        
        years = [year for i in range(size)]

        quick_pmp = [1 for i in range(size)] 
        hourly_output = [0 for i in range(size)] 

        
        # combine
        # year, rs, temp_mean, temp_min, temp_max, wind_speed, lon, lat, 
        input_para = np.array(list(zip(pixel_type.reshape(-1).T, 
                               years, 
                               rsds.values.reshape(rsds.shape[0],-1).T, 
                               wind_speed.values.reshape(wind_speed.shape[0],-1).T, 
                               tas.values.reshape(tas.shape[0],-1).T, 
                               tasmin.values.reshape(tasmin.shape[0],-1).T, 
                               tasmax.values.reshape(tasmax.shape[0],-1).T,                         
                               lons.reshape(-1).T, 
                               lats.reshape(-1).T,
                               quick_pmp,
                               hourly_output                                                        
                           )), dtype=object)

        del tasmax, tasmin, tas, wind_speed, rsds
        gc.collect()    
        print(year, 'finished: combine data', datetime.datetime.now())

            
        t = datetime.datetime.now()
        r = multi_process(input_para)
        print(datetime.datetime.now() - t )
        input_para = None
        
        maps = np.array(r, dtype = np.float32).reshape(len(lat), len(lon), len(time))   
        r = None
        
        maps = xr.DataArray(maps, dims = ['lat','lon','time'], name = 'pv_power_production',
                            coords={'lat': lat,'lon': lon, 'time':time})
        
        maps.attrs['units'] = 'Wh'       
        maps.to_netcdf(r'./daily_power_production/pv/global_pv_power_' + scenario[s] + '_' + str(year) + '_' + climate_models +'.nc', 
                       encoding={'pv_power_production': {'zlib': True, 'complevel': 6}}) 
        del maps
        print('finished output', climate_models, scenario[s], year)



