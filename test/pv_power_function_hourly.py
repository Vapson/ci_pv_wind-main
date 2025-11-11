# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import pvlib
import warnings


class PVsystem:
    '''
    input parameters:  
        year: int    
        time: numpy.datetime64                                                             
        rs：J/m2, houly solar radiation array 
        temp_air: air temperature, 2m above surface  K
        wind_speed: m/s
        lon: degree  (-180 - +180)    
        lat: degree  (-90-90)  

    '''    
    
    def __init__(self, year, time, rs, temp_air, wind_speed, lon, lat):               
        self.rs = rs   
        self.temp_air = temp_air - 273.15
        self.wind_speed = wind_speed        
        self.lon = lon
        self.lat = lat

        
        self.year = year   
        self.timezone = self.TimeZone()
        self.utc_time = pd.to_datetime(time).tz_localize(None).tz_localize("UTC")
        self.local_time = self.utc_time.tz_convert(self.timezone)
        self.local_doy = self.local_time.dayofyear
        self._module = pvlib.pvsystem.retrieve_sam('CECMod').Jinko_Solar_Co___Ltd_JKM410M_72HL_V
        
      
          
    def TimeZone(self):
        #function: Divide the time zone by longitude
        lon = self.lon
        
        if lon < -180:
            lon = -180
        if lon > 180:
            lon -= 360      
            
        timezone = round(lon / 15)
        hours = int(timezone)         

        utc_str = f"UTC{hours:+03d}:00"             
        return utc_str  
    
   

    def HourAngle(self):   
        # function: Calculate the solar hour Angle
        # return: solar hour angle (degree)
               
        # eot--> Soteris A. Kalogirou, "Solar Energy Engineering Processes and Systems, 
        #        2nd Edition" Elselvier/Academic Press (2009).        
        eot = pvlib.solarposition.equation_of_time_pvcdrom(self.local_doy)/60 # min --> hour        
        
        #
        # lon in pvlib input format: -180 - +180
        t = int(self.timezone[3:6])*15 # The central longitude of the corresponding time zone
        w = self.local_time.hour + 4* (self.lon - t) / 60 + eot # It takes the 4 minutes to rotate 1 degree.
        ha = (w -12) * 15
        return np.array(ha, dtype=np.float32)
    
    
    
    def Sunpath(self):   
        # function: Calculate the zenith, elevation, equation_of_time        
        # lon in pvlib input format: -180 - +180
        ephem_data = pvlib.location.Location(self.lat, self.lon).get_solarposition(self.local_time)
        return ephem_data
    
    
    
    
    def SunTime(self):  
        # function: Calculate the sunrise and sunset time (hour)
        with warnings.catch_warnings():
             warnings.simplefilter('ignore')            
             suntime = pvlib.solarposition.sun_rise_set_transit_spa(self.local_time, self.lat, self.lon)

        sunrise_hour=np.array( [np.nan if pd.isnull(suntime['sunrise'].iloc[x]) else suntime['sunrise'].iloc[x].timetuple().tm_hour
                               for x in range(len(suntime))] )
        sunset_hour=np.array( [np.nan if pd.isnull(suntime['sunset'].iloc[x]) else suntime['sunset'].iloc[x].timetuple().tm_hour
                               for x in range(len(suntime))] )
        return sunrise_hour, sunset_hour   
    
    
    
    def dailyExtraterrestrialRadiation(self):
        ISC = 1366.1 # solar constant (W/m2)
        lat = self.lat          
    
        # declination angle
        da = pvlib.solarposition.declination_spencer71(self.local_doy) 
        
        # sunset hour angle
        x=np.array(-np.tan(lat/180*np.pi)*np.tan(da))
        x[x>1]=1
        x[x<-1]=-1
        ws = np.arccos(x) *180 /np.pi
                  
        E0 = 1 + 0.033 * np.cos(2 * np.pi * self.local_doy/365)
        I0 = 24*3600/np.pi * ISC * E0 *( np.cos(lat/180*np.pi) * np.cos(da) *  np.sin(ws/180*np.pi)
                                        + np.pi*ws/180 * np.sin(lat/180*np.pi)*np.sin(da) )

        return np.array(I0, dtype=np.float32)
        
    
    
    def HourlyExtraterrestrialRadiation(self):
        ISC = 1366.1 #solar constant (W/m2)
        lat = self.lat
        
        # declination angle (radians)
        da = pvlib.solarposition.declination_spencer71(self.local_doy) 
        
        # hour angle
        HourAngles = self.HourAngle()
        w1 = HourAngles
        w2 = np.hstack((HourAngles[1:],HourAngles[-1:]+15))

        error = ((w2-w1)<14) | ((w2-w1)>16)
        w2[error] = w1[error]+15
        
        E0 = 1 + 0.033 * np.cos(2*np.pi*self.local_doy/365)
        I0 = 12 * 3600/np.pi * ISC * E0 *( np.cos(lat/180*np.pi) * np.cos(da) * ( np.sin(w2/180*np.pi) - np.sin(w1/180*np.pi))
                                         + np.pi*(w2-w1)/180 * np.sin(lat/180*np.pi)*np.sin(da) )
            
        return np.array(I0, dtype=np.float32)
    

    
    
    def kd(self):
        # function: using BRL model to caculate the fraction of diffuse solar radiation from global solar radiation     
        # Lauret et al.2013 Bayesian statistical analysis applied to solar radiation modelling. 
        beta0 = -5.32 
        beta1 = 7.28
        beta2 = -0.03
        beta3 = -0.0047 
        beta4 = 1.72 
        beta5 = 1.08 
        
        
        rs = self.rs 
        sunrise_hour,sunset_hour = self.SunTime()

        I0 = self.dailyExtraterrestrialRadiation()
        I0[I0==0] = 1 
        #
        # daily clearness index
        daily_rs = rs.reshape(24, int(len(rs)/24))
        daily_rs = np.sum(daily_rs,axis=0)

        hourly_d_rs=[]
        for i in range(len(daily_rs)):
            hourly_d_rs.extend([daily_rs[i]]*24)
        Kt = np.array(hourly_d_rs)/I0
        
        # hourly clearness index
        kt  = rs / self.HourlyExtraterrestrialRadiation()
        kt[kt>1] = 1
        kt[kt<0] = 0

        # apparent solar time
        AST = (self.HourAngle()/15) + 12
        
        # solar altitude angle
        alpha = np.array(self.Sunpath()['apparent_elevation'])
        
        # persistence of the sky conditions
        timezone = self.TimeZone()
        utcoffset = int(timezone[4:6])
        indexs = np.arange(utcoffset, utcoffset+len(rs),1,dtype=np.int32)
        hour_index = indexs%24 
        phi = np.convolve(kt, np.array([1/2, 1/2]), mode='full')[:-1]
        

        phi[hour_index==sunrise_hour] = kt[hour_index==sunrise_hour]
        phi[hour_index==sunset_hour] = kt[hour_index==sunset_hour]
        phi[hour_index<sunrise_hour] = 0
        phi[hour_index>sunset_hour] = 0
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k = 1/(1+np.exp(beta0 + beta1*kt + beta2*AST + beta3*alpha + beta4*Kt + beta5*phi))             
        return k    
    

    

    def Fixed(self):
        # funciton: no tracking system (panel tilt, azimuth: south)
        # panel azimuth angle = 180 ,facing South
        # tile: plane tilt angle 
        # dni:Direct Normal Irradiance. [W/m2]
        # ghi:Global horizontal irradiance. [W/m2]
        # dhi:Diffuse horizontal irradiance. [W/m2]

        k = self.kd()
        rs = self.rs
        lat = self.lat
        
        Sunpath = self.Sunpath()
        solar_zenith_angle = Sunpath['apparent_zenith']
        solar_azimuth_angle = Sunpath['azimuth']

        ghi = rs / 3600 
        dhi = rs * k / 3600 
        dni = pvlib.irradiance.dni(ghi, dhi, np.array(solar_zenith_angle))
        dni[pd.isna(dni)] = 0            
        
        if lat>=0:
            panel_azimuth = 180 #facing south
            # source: World estimates of PV optimal tilt angles and ratios of sunlight incident
                      #upon tilted and tracked PV panels relative to horizontal panels.2018
            tilt = 1.3793+lat*(1.2011+lat*(-0.014404+lat*0.000080509))
        else:
            panel_azimuth = 0 #facing north
            tilt = -0.41657+lat*(1.4216+lat*(0.024051+lat*0.00021828))
            tilt = -tilt     
            
        ipoa = pvlib.irradiance.get_total_irradiance( tilt,
                                                       panel_azimuth, #facing south
                                                       solar_zenith_angle, solar_azimuth_angle,            
                                                       dni=dni,ghi=ghi,dhi=dhi,model='king')     
        aoi = pvlib.irradiance.aoi(tilt,panel_azimuth,solar_zenith_angle, solar_azimuth_angle)
        ipoa['aoi'] = aoi
        return ipoa
    
     
  
    def PVSystem_sapm_celltemp(self, poa_global):
        # function: cell temperature
        a, b, deltaT = (-2.98, -0.0471, 1)  
        temp_air = self.temp_air
        wind_speed = self.wind_speed
        temp_cell = pvlib.temperature.sapm_cell(poa_global, temp_air, wind_speed, a, b, deltaT)
        return temp_cell
    
 
    
    def CECmod(self):
        Fixed = self.Fixed()
        iam_modifier = pvlib.iam.physical(Fixed['aoi'])
        module = self._module
        fd = module.get('FD', 1.0)    
        effective_irradiance = 1*(Fixed['poa_direct'] * iam_modifier +
                                    fd * Fixed['poa_diffuse'])    
        temp_cell = self.PVSystem_sapm_celltemp(Fixed['poa_global'])

        IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_cec(
            effective_irradiance, temp_cell,
            alpha_sc=module['alpha_sc'],
            a_ref=module['a_ref'],
            I_L_ref=module['I_L_ref'],
            I_o_ref=module['I_o_ref'],
            R_sh_ref=module['R_sh_ref'],
            R_s=module['R_s'],
            Adjust=module['Adjust']
        )
    
        power_dc = pvlib.pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)['p_mp']  
        return power_dc 

    

    
    def Area_ajust(self):
        Area_ajust=  1 / self._module.A_c               
        return Area_ajust    
    
    

    
# In[1]
def main(inputs):
    land, year, time, rsds, wind, temp, lon, lat = inputs[:]
    if land==1:
        f=PVsystem(year,
                   time,
                   rsds,
                   temp,
                   wind, 
                   lon,
                   lat)

        power = f.CECmod() * f.Area_ajust() 
        return power#np.array(power, dtype=np.float32)
    
    else:
        power=np.array([0]*len(rsds),dtype=np.float32)
        return power




