# -*- coding: utf-8 -*-
import os
os.chdir(r'...')
import numpy as np
import pandas as pd
import datetime
import xarray as xr
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')


def renumber_continuous(nums):
    # Generate event number
    nums = np.array(nums)
    diffs = np.diff(nums)
    groups = np.cumsum( np.concatenate(([1], diffs != 1)) )
    return groups



def process_chunk(chunk):    
    group_keys = ['event_ID', 'lon_index', 'lat_index']
    
    # duration
    du = chunk.groupby(group_keys).size().reset_index(name='duration_days')
    
    # intensity
    d_power = chunk.groupby(group_keys)[['power_production', 'ave', 'thrs_10th']].sum().reset_index()
    
    # combine
    result = pd.merge(du, d_power, on=group_keys)
    return result



def reconstruct_2d_array(index_value_pairs, num_rows, num_cols):
    rows, cols, values = zip(*index_value_pairs)
    return coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).toarray()




def main(power, climate_model, scenario, year, thresholds, clim_aves):
    # Generate the event information table and annual maps
    #
    # define save path and name    
    save_path = os.path.join(r'./ELP_event/maps', power+'_new') 
    save_name = 'ELPevent_info_' +  scenario + '_' + str(year) + '_' + climate_model + '.nc'
    
    if os.path.exists(os.path.join(save_path, save_name)):
        print('file exits')
        

    else:
        # Set path
        if power == 'wind':
            file_path = r"./daily_power_production/wind/land_and_ocean"

        if power == 'pv': 
            file_path = r"./daily_power_production/pv"

        #        
        # read daily power generation curve  
        file_path = os.path.join(file_path) 
        file_name = 'global_' + power + '_power_' + scenario + '_' + str(year) + '_' + climate_model + '.nc'

        daily_data = xr.open_dataset( os.path.join(file_path, file_name) )[f'{power}_power_production']
        lat, lon = daily_data.lat.values, daily_data.lon.values

        
        pp = daily_data.values          
        days = len(daily_data['time'])
        if days == 366:
            pp = pp[:,:,~((daily_data['time'].dt.month==2) & (daily_data['time'].dt.day==29))]
            days = 365


        # Generate time, latitude and longitude indexes        
        lat_length, lon_length = len(daily_data.lat), len(daily_data.lon)
        doy = np.arange(0, days, 1)
        lat_index = np.arange(0, lat_length, 1)
        lon_index = np.arange(0, lon_length, 1)
        lons_index, lats_index, doys  = np.meshgrid(lon_index, lat_index, doy)   

 

        # identify events
        judge = np.where((pp<thresholds), 1, 0)
        lon_index = lons_index[judge==1]
        lat_index = lats_index[judge==1]
        doy_index = doys[judge==1]
        elp_event = pp[judge==1] 
        ave = clim_aves[judge==1]
        thr = thresholds[judge==1]


        # convert to dataframe
        df = np.concatenate((lon_index.reshape(-1,1), 
                             lat_index.reshape(-1,1), 
                             doy_index.reshape(-1,1), 
                             ave.reshape(-1,1),
                             thr.reshape(-1,1),
                             elp_event.reshape(-1,1)),
                            axis=1, dtype=np.float32) 

        df = pd.DataFrame(df, columns = ['lon_index',
                                         'lat_index',
                                         'doy',
                                         'ave',
                                         'thrs_10th',#'thrs_5th','thrs_1th',
                                         'power_production'])   
        
        del lon_index, lat_index, ave, thr, doy_index, elp_event, thresholds, clim_aves, doys



        # remove_seasonal_no_power
        # Data with 10 quantile < 0 is deleted        
        df = df[(df['thrs_10th']>1e-5)&(df['ave']>1e-5)]
        df.reset_index(drop = True, inplace = True)     

        # ignore power consumption during the night
        df['power_production'][df['power_production']<0] = 0 


        # add event_id
        df = df.sort_values(by=['lon_index', 'lat_index']).reset_index(drop=True)
        df['event_ID'] = renumber_continuous(df['doy'])       
        
        # get events
        extreme_event = process_chunk(df)
        extreme_event['power_gap'] = extreme_event['ave'] - extreme_event['power_production']
        extreme_event['intensity_re'] = extreme_event['power_gap'] / extreme_event['ave']

        extreme_event = extreme_event[extreme_event['duration_days']<360].reset_index(drop=True)
        print(extreme_event['duration_days'].max())
        del df        


        #
        # ----  save event information table  
        # np.float16 will be failed to save the event ID, but will save memory          
        if power == 'wind':            
            df = np.concatenate((extreme_event.iloc[:,:4],
                                   extreme_event.iloc[:,4:8]/1e+4, # --> W --10**4 W, 
                                   extreme_event.iloc[:,8:],
                                   ), axis=1, dtype=np.float16
                                  ) 

        else:
            df = np.array(extreme_event, dtype=np.float16)
            
        if np.isinf(np.max(df[:,1:])):
            df = np.array(extreme_event, dtype=np.float32)

        #np.save(os.path.join(save_path, save_name), df)
        print('finished: save event.npy', power, climate_model, scenario, year)
  

        #
        # ----  generate annual extreme low production map   
        ds = extreme_event.groupby(['lon_index', 'lat_index']).sum().reset_index()
        dm = extreme_event.groupby(['lon_index', 'lat_index']).mean().reset_index()    
        dc = extreme_event.groupby(['lon_index', 'lat_index']).count().reset_index()  
        dmax = extreme_event.groupby(['lon_index', 'lat_index']).max().reset_index()
        
        # elp
        index_value_pairs = list(zip(ds['lat_index'].astype(int), 
                                     ds['lon_index'].astype(int), 
                                     ds['power_gap']))       
        elp =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length) 

        # elp-max
        index_value_pairs = list(zip(dmax['lat_index'].astype(int), 
                                     dmax['lon_index'].astype(int), 
                                     dmax['power_gap']))       
        elp_max =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length) 
        
        # total days
        index_value_pairs = list(zip(ds['lat_index'].astype(int), 
                                     ds['lon_index'].astype(int), 
                                     ds['duration_days']))       
        total_days =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length) 

        
        # average duration_days
        index_value_pairs = list(zip(dm['lat_index'].astype(int), 
                                     dm['lon_index'].astype(int), 
                                     dm['duration_days']))       
        duration =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length) 

        
        # max duration_days
        index_value_pairs = list(zip(dmax['lat_index'].astype(int), 
                                     dmax['lon_index'].astype(int), 
                                     dmax['duration_days']))       
        duration_max =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length)   
        

        # frequency
        index_value_pairs = list(zip(dc['lat_index'].astype(int), 
                                     dc['lon_index'].astype(int), 
                                     dc['event_ID']))       
        frequency =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length)  




        # average intensity - absolute
        index_value_pairs = list(zip(dm['lat_index'].astype(int), 
                                     dm['lon_index'].astype(int), 
                                     dm['power_gap']))       
        intensity_ab =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length)  



        # average intensity - relative
        index_value_pairs = list(zip(dm['lat_index'].astype(int), 
                                     dm['lon_index'].astype(int), 
                                     dm['intensity_re']))       
        intensity_re =  reconstruct_2d_array(index_value_pairs, lat_length, lon_length) 


        #
        maps = np.concatenate(( elp.reshape(1, lat_length, lon_length), 
                                elp_max.reshape(1, lat_length, lon_length),
                                total_days.reshape(1, lat_length, lon_length), 
                                duration.reshape(1, lat_length, lon_length), 
                                duration_max.reshape(1, lat_length, lon_length), 
                                frequency.reshape(1, lat_length, lon_length), 
                                intensity_ab.reshape(1, lat_length, lon_length),
                                intensity_re.reshape(1, lat_length, lon_length)),
                              axis=0, dtype=np.float32) 

        var_names = ['ELP', 'ELP_max', 'total_days', 'duration_ave', 'duration_max', 'frequency', 'intensity_ab_ave', 'intensity_re_ave']
        maps = xr.Dataset(
                            {var_names[i]: (["lat", "lon"], maps[i,:,:]) for i in range(len(var_names))}, 
                            coords={
                                    'lat': lat,
                                    'lon': lon
                                    }
                            )  
    
        
        if power == 'wind':
            maps.attrs['unit_for_absolute_value'] = 'W'  

        if power == 'pv':
            maps.attrs['unit_for_absolute_value'] = 'Wh' 
        
        
        save_path = './ELP_event/maps'
        file_name = 'ELPevent_info_' +  scenario + '_' + str(year) + '_' + climate_model + '.nc'

        maps.to_netcdf(os.path.join(save_path, power+'_new', file_name),
                       encoding={var_names[i]: {'zlib': True, 'complevel': 6} for i in range(len(var_names))})   
        print('finished: save map.nc', power, climate_model, scenario, year)

    


            
'''
#
# main line
#
power = 'pv'
scenarios = ['ssp126', 'ssp245', 'ssp370']
start_years = [2015, 2015, 2015]
end_years = [2100, 2100, 2100]


climate_models = [
#                  'ACCESS-ESM1-5',
#                   'BCC-CSM2-MR', 
#                  'CanESM5',
#                  'CMCC-ESM2', 'EC-Earth3',
#                  'FGOALS-g3', 
                    'GFDL-ESM4', 
#                  'INM-CM4-8', 
#                  'IPSL-CM6A-LR',  'MIROC6',
#                  'CNRM-ESM2-1',
#                   'GISS-E2-1-G',
#                   'KACE-1-0-G',
#                  'MPI-ESM1-2-HR', 
#                  'UKESM1-0-LL',
#                  'MRI-ESM2-0', 
#                  'NorESM2-MM', 
                 ] 

for c in range(0, len(climate_models)):    
    climate_model = climate_models[c]
    
    # -------- baseline
    scenario = 'historical'
    start_year = 1985
    end_year = 2014
    time_range = np.arange(start_year, end_year+1)  
    
    for year in time_range:        
        # read thresholds & clim_ave
        thrs_name = r'./ELP_event/threshold/threshold_for_baseline/' + power + '_threshold_' + climate_model + '_ws15_r' + str(year) + '_remove_seasonal_cycle.nc'
        thresholds = xr.open_dataset(thrs_name)
        thresholds = thresholds['threshold_10th_1985-2014'].values 
        # --> (lat, lon, time)
        thresholds = np.transpose(thresholds, (1, 2, 0))

        ave_name =  r'./ELP_event/threshold/threshold_for_baseline/' + power + '_clim_ave_' + climate_model + '_ws15_r' + str(year) + '_remove_seasonal_cycle.nc'
        clim_aves = xr.open_dataset(ave_name)
        clim_aves = clim_aves['clim_ave_1985-2014'].values
        # --> (lat, lon, time)
        clim_aves = np.transpose(clim_aves, (1, 2, 0))  

        t = datetime.datetime.now()
        main(power, climate_model, scenario, year, thresholds, clim_aves)
        print('finished:', climate_model, scenario, year, datetime.datetime.now()-t) 
        
        
        
    # -------- future
    
    # read thresholds & clim_ave
    thrs_name = r'./ELP_event/threshold/threshold_for_future/' + power + '_threshold_' + climate_model + '_ws15_remove_seasonal_cycle.nc'
    thresholds = xr.open_dataset(thrs_name)
    thresholds = thresholds['threshold_10th_1985-2014'].values 
    # --> (lat, lon, time)
    thresholds = np.transpose(thresholds, (1, 2, 0))

    ave_name =  r'./ELP_event/threshold/threshold_for_future/' + power + '_clim_ave_' + climate_model + '_ws15_remove_seasonal_cycle.nc'
    clim_aves = xr.open_dataset(ave_name)
    clim_aves = clim_aves['clim_ave_1985-2014'].values
    # --> (lat, lon, time)
    clim_aves = np.transpose(clim_aves, (1, 2, 0))  
    

    for s in range(0, len(scenarios)):
        scenario = scenarios[s]
        start_year = start_years[s]
        end_year = end_years[s]
        time_range = np.arange(start_year,end_year+1)  
        
        for year in time_range:
            t = datetime.datetime.now()
            main(power, climate_model, scenario, year, thresholds, clim_aves)
            print('finished:', climate_model, scenario, year, datetime.datetime.now()-t)
    
'''      

            


        
