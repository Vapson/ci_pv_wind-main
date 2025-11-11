# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import windpowerlib as wt
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import weibull_min
import scipy.special as sp

#In[1] CLASS_function
class windpower:
    '''
    input parameters
    ----------
        tas: daily mean air temperature at 2m height (K)                                               
        dem：m      
        wind_speed: daily mean wind speed at 10m height (m/s)
        pl_exponent: power law exponent
        
    '''    
    def __init__(self, pixel_type, Tas, dem, wind_speed, pl_exponent):
        self.tas = Tas
        self.dem = dem
        self.wind_speed = wind_speed
        self.wind_speed_height = 10 # unit: m        
        self.pl_exponent = pl_exponent
        self.k = 2 
        self.pixel_type = pixel_type
        
        if self.pixel_type == 1:
            # onshore
            # from Wind turbine: V112/3075
            self.hub_height = 94 # unit: m
            enercon_e126 = {
                    'turbine_type': 'V112/3075',  # turbine type as in oedb turbine library
                    'hub_height': self.hub_height  # in m
                }
            # initialize WindTurbine object
            e126 = wt.WindTurbine(**enercon_e126)
            self.power_curve = e126.power_curve  
            self.cut_in_ws = 3
            self.cut_out_ws = 25
            # self.power_coefficient_curve = e126.power_coefficient_curve
            self.rotor_radius = e126.rotor_diameter / 2 # unit: m,             
            
            
        if self.pixel_type == 2:
            # offshore
            # from Wind turbine: V164/9500
            self.hub_height = 105 # unit: m
            enercon_e126 = {
                    'turbine_type': 'V164/9500',  # turbine type as in oedb turbine library
                    'hub_height': self.hub_height  # in m
                }
            # initialize WindTurbine object
            e126 = wt.WindTurbine(**enercon_e126)
            self.power_curve = e126.power_curve  
            self.cut_in_ws = 3.5
            self.cut_out_ws = 25
            # self.power_coefficient_curve = e126.power_coefficient_curve            
            self.rotor_radius = e126.rotor_diameter / 2 # unit: m, 
            



    def Pb(self):
        r"""
        Calculates the air pressure at globe surface.
        
        unit: Pa
        """
        P0 = 101325 #std Pa
        M = 0.02896 #kg/mol
        g = 9.807 #m/s2
        R = 8.3143 #(N*m)/(mol*K)   
        return P0 * np.exp( - M * g / (R * self.tas) * self.dem)
    
    
    
    def get_temp_at_hub_height(self):
        r"""
        Calculates the temperature at hub height using a linear gradient. (windpowerlib
    
        A linear temperature gradient of -6.5 K/km (-0.0065 K/m) is assumed. This function is
        carried out when the parameter `temperature_model` of an instance of
        the :class:`~.modelchain.ModelChain` class is
        'temperature_gradient'.
    
        Parameters
        ----------
        temperature : :pandas:`pandas.Series<series>` or numpy.array
            Air temperature in K.
        temperature_height : float
            Height in m for which the parameter `temperature` applies.
        hub_height : float
            Hub height of wind turbine in m.
    
        Returns
        -------
        :pandas:`pandas.Series<series>` or numpy.array
            Temperature at hub height in K.
        Ref:
            windpowerlib      
        """
        return self.tas - 0.0065 * self.hub_height
    
    
    def get_air_density(self):
        r"""
        Calculates the density of air at hub height using the ideal gas equation.
    
    
        Parameters
        ----------
        pressure : :pandas:`pandas.Series<series>` or numpy.array
            Air pressure in Pa.
        pressure_height : float
            Height in m for which the parameter `pressure` applies.
        hub_height : float
            Hub height of wind turbine in m.
        temperature_hub_height : :pandas:`pandas.Series<series>` or numpy.array
            Air temperature at hub height in K.
    
        Returns
        -------
        :pandas:`pandas.Series<series>` or numpy.array
            Density of air at hub height in kg/m³.
            Returns a pandas.Series if one of the input parameters is a
            pandas.Series.
        Ref:
            windpowerlib    
        """
        pressure = self.Pb()
        temperature_hub_height = self.get_temp_at_hub_height()
        return (
            (pressure / 100 - self.hub_height * 1 / 8)
            * 100
            / (287.058 * temperature_hub_height)
        )
    
  

    def wind_speed_hub_height(self):
        r"""
        Calculates the wind speed at hub height using the hellman equation.
    
        It is assumed that the wind profile follows a power law. 
        """
        return self.wind_speed * (self.hub_height / self.wind_speed_height) ** self.pl_exponent
    
    
   
    
    def get_weibull_pdf(self):
        ws_hub = self.wind_speed_hub_height()
        
        lambda_param = ws_hub / sp.gamma(1 + 1/self.k)   

        # Generate the instantaneous wind speed probability density function of the Weibull distribution 
        U = np.linspace(self.cut_in_ws, self.cut_out_ws, 500)
        if np.isscalar(ws_hub):
            weibull_pdf = weibull_min.pdf(U, c = self.k, scale = lambda_param)  
        else:
            U = U[:, np.newaxis]  # shape: (500, 1)
            weibull_pdf = weibull_min.pdf(U, c = self.k, scale = lambda_param)  # shape: (500, n)
        return weibull_pdf, U
    

        

    def get_daily_mean_wind_power(self): 
        r"""
        Calculates the daily mean wind power production using target wind turbine. 
        
        unit: W
        """        
        # Calculate the probability density of wind speed
        pdf_values, U = self.get_weibull_pdf()         
        
        curve_ws, curve_power = self.power_curve['wind_speed'], self.power_curve['value']
        
        U_c = U * (self.get_air_density() / 1.225)**(1/3)
        power_values = np.interp(U_c, curve_ws, curve_power)

        # np.trapz：Use the Trapezoidal Rule for numerical integration.
        # It integrates the power-probability density product within the U range to obtain the weighted expected power.
        expected_power = np.trapz(power_values * pdf_values, U, axis=0) 
        return expected_power
        
        
       
 
        
# In[]
def main(input_para):
    # pixel_type = 0: none
    # pixel_type = 1: onshore
    # pixel_type = 2: offshore
    pixel_type, tas, dem, wind_speed, pl_exponent = input_para[:]
    if pixel_type != 0:
        f = windpower(pixel_type, tas, dem, wind_speed, pl_exponent)
        energy = f.get_daily_mean_wind_power() 
        return np.array(energy, dtype=np.float32)
    else:
        return np.array([0]*len(wind_speed),dtype=np.float32)



