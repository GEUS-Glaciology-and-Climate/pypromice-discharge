# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:06:20 2024

@author: rabni
"""


import pandas as pd 
import numpy as np
import xarray as xr
import toml, os
import glob
from pathlib import Path
from datetime import timedelta, datetime
from typing import Sequence, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
def process(inpath, config_file, air_config=None, air_inpath=None,l1=False):
    '''Perform Level 0 to Level 3 processing'''
    # assert(os.path.isfile(config_file))
    # assert(os.path.isdir(inpath))
    
    # Load config, variables CSF standards, and L0 files
    config = get_config(config_file, inpath)

    # Load L0 data files
    ds_list = get_l0(config)
    
    # Load L0 atmospheric data
    if air_config!=None and air_inpath!=None:
        assert(os.path.isfile(air_config))
        assert(os.path.isdir(air_inpath))
        aconfig = get_config(air_config, air_inpath)
        l0_air = get_air(aconfig)
    else:
        l0_air=None
    
    # Perform processing
    
    if not l1:
        l1 = get_l1(ds_list, config, l0_air)
        l2 = get_l2(l1)
        l3 = get_l3(l2)
        return l3
    else:
        l1 = get_l1(ds_list, config, l0_air,cor=False)
        return l1
def get_l0(config):
    '''Get L0 discharge data from config file'''
    ds_list=[]
    for i, c in config.items(): 
        print(f'Loading {c["file"]}')
        ds = load_l0(c['file'], 
                     c['columns'],
                     c['skiprows'],
                     c['nodata'],
                     c['format'],
                     c['last_good_data_line'])
        ds_list.append(ds)
    return ds_list
    
def get_air(config):  
    '''Get atmospheric data from config file'''
    for i, c in config.items():
        print(f'Loading external atmospheric data from {c["file"]}...')
        met_df = pd.read_csv(c['file'], 
                         comment='#', 
                         sep=';',
                         names=c['columns'], 
                         skiprows=c['skiprows'], 
                         na_values= c['nodata'],
                         skip_blank_lines=True, 
                         engine='python'
                         )
    t=[]
    for y,m,d,h in zip(list(met_df['year']), list(met_df['month']), 
                       list(met_df['day']), list(met_df['hour'])):
        t.append(datetime(int(y), int(m), int(d), int(h)))
    met_df['time']=t
    met_df = met_df.set_index('time')
    met_df.drop(columns=['year','month','day','hour', 'SKIP_6'], inplace=True)
    
    # Drop SKIP columns
    for d in met_df.columns:
        if 'skip' in d.lower():
            met_df.drop(columns=d, inplace=True)
            
    met_ds = xr.Dataset.from_dataframe(met_df)   
    return met_ds
        
def write_csv(ds, outfile):
    '''Write Dataset to .csv file'''
    df = ds.to_dataframe()
    df.to_csv(outfile) 
   

def write_netcdf(ds, outfile):
    '''Write Dataset to .nc file'''
    ds.to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)             
    
"""
def write_netcdf(ds, filename):
    '''Write Dataset to .nc file'''
    
    meta = pd.read_csv("nc_var_meta.csv")
    title_name = meta['title']
    names = meta["names"] 
    longnames = meta["long_names"]
    units = meta["units"]
    ds_out = nc.Dataset(filename, 'w', format='NETCDF4')
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    time_dim = ds_out.createDimension('time', len(ds.time))
    
    time_var = ds.createVariable('time', np.datetime64, ('time'), zlib=True)
    time_var.units = 'degrees_north'
    time_var.standard_name = 'time in YYYY-MM-DDTHH:MM:SS'
    time_var.axis = ''
    time_var[:] = ds.time

    ds_out.title = f"Hourly Hydrological Monitoring  at {title_name}, promice_discharge v. 2.1"
    ds_out.summary = ''
    ds_out.keywords = 'Cryosphere > Land Ice > Land Ice Albedo > Reflectance > Greenland > Northern Hemisphere > Grain Size'
    ds_out.instrument = "OLCI"
    ds_out.platform = "Sentinel-3A"
    ds_out.start_date_and_time = date + "T08:00:00Z"
    ds_out.end_date_and_time = date + "T16:00:00Z"
    ds_out.naming_authority = "geus.dk"
    
    ds_out.summary = ''
    ds_out.keywords = 'Cryosphere > Land Ice > Land Ice Albedo > Reflectance > Greenland > Northern Hemisphere > Grain Size'
    ds_out.activity = 'Space Borne Instrument'
    ds_out.geospatial_lat_min = lat_min
    ds_out.geospatial_lat_max = lat_max
    ds_out.geospatial_lon_min = lon_min
    ds_out.geospatial_lon_max = lon_max
    ds_out.time_coverage_start = date + "T08:00:00Z"
    ds_out.time_coverage_end = date + "T16:00:00Z"
    ds_out.history = current_date + ' processed'
    ds_out.date_created = current_date
    ds_out.creator_type = "group"
    ds_out.creator_institution = "Geological Survey of Denmark and Greenland (GEUS)"
    ds_out.creator_email = " jeb@geus.dk, bav@geus.dk, rabni@geus.dk,adrien.wehrle@geo.uzh.ch"
    ds_out.creator_name = "Jason Box, Baptiste Vandecrux, Rasmus Bahbah Nielsen, Adrien Wehrlé"
    ds_out.creator_url = "https://orcid.org/0000-0003-2342-639X"
    ds_out.institution = "Geological Survey of Denmark and Greenland (GEUS)"
    ds_out.publisher_type = "Institute"
    ds_out.publisher_name = "Geological Survey of Denmark and Greenland (GEUS), Glaciology and Climate Department"
    ds_out.publisher_url = "geus.dk"
    ds_out.publisher_email= "jeb@geus.dk"
    ds_out.project = "Operational Sentinel-3 snow and ice products (SICE)"
    ds_out.license = "None"
    
    for v in ds: 
        if v in names:
            idx = names.index(v)
            z_out = ds.createVariable(v, 'f4', ('time'),zlib=True)
            z_out[:] = ds[v].to_numpy()
            z_out.standard_name = v
            z_out.long_name = longnames[idx]
            z_out.units = units[idx]
            
"""    
def get_l1(l0_list, config, l0_air=None,cor=True):
    '''Perform L0 to L1 processing, where input is from a list of Dataset objects
    and corresponding config toml file'''
    ds_list=[]
    
    for i,ds in enumerate(l0_list):
        
        c = list(config.values())[i]
        print(f'Correcting variables in {c["file"]}...')
        
        # Reformat all variables to numerical values
        for l in list(ds.keys()):
            if l not in ['time']:
                ds[l] = reformat_array(ds[l])
            
        # Apply time offset
        ds['time'] = offset_time(ds['time'], c['utc_offset'])
        
        if cor:
            if 'analog_dvr_scale' in c: 
                if 'h_l_h' in ds:
                    ds['h_l_h'] = ds['h_l_h'] * 10**-2 # Going from centimeters water level to meters
                elif 'h_u_h' in ds:
                    ds['h_u_h'] = ds['h_u_h'] * 10**-2 # Going from centimeters water level to meters
            
            # Calculate upper and lower dH
            if 'dh_bolt_u' in c and 'dh_diver_u' in c:
                h_u_dvr = calc_dh(c['dh_bolt_u'], c['dh_diver_u'], c['dH_0']) 
                ds['h_u_dvr'] = (('time'), [h_u_dvr]*len(ds['time'].values))
                
            if 'dh_bolt_l' in c and 'dh_diver_l' in c:            
                h_l_dvr = calc_dh(c['dh_bolt_l'], c['dh_diver_l'], c['dH_0']) 
                ds['h_l_dvr'] = (('time'), [h_l_dvr]*len(ds['time'].values))               
            # Mask out where p_wtr_u and p_wtr_l are nan to also be nan here
            
            # Apply pressure offset, also apply to p_wtr_l, p_air
            if hasattr(ds, 'p_wtr_u'): 
                ds['p_wtr_u_cor'] = offset_press(ds['p_wtr_u'], c['p_offset_u']) 
                ds['p_wtr_u_cor'] = ds['p_wtr_u_cor'].where(ds['p_wtr_u_cor'] != 1450.0)
            if hasattr(ds, 'p_wtr_l'):
                ds['p_wtr_l_cor'] = offset_press(ds['p_wtr_l'], c['p_offset_l'])
            if hasattr(ds, 'p_air'):
                ds['p_air_cor'] = offset_press(ds['p_air'], c['p_offset_a'])
  
        # Resample to hourly mean values
        ds = resample_data(ds, '60min')
        ds_list.append(ds) 

    # Combine all files
    print('Combining files into single L1 object...')
    l1 = ds_list[0]
    for d in ds_list[1:]:
        l1 = l1.combine_first(d)             
  
    if l0_air != None:
        print('Merging external atmospheric data...')
        
        #l1['t_air_dmi'] = l0_air
        #l1['t_air_comb'] = xr.merge([l1['t_air_baro'],l0_air],compat='override')
        
        l1 = xr.merge([l1,l0_air],compat='override')
        
        
    
    print('L1 processing complete')
    return l1
        
def get_l2(L1):
    '''Perform L1 to L2 processing'''
    ds = L1.copy(deep=True)   
    print('Calculating water level...')
        
    # Checking if there is any water level data in the raw data
    if 'h_l_h' in ds:
        raw_l2 = ds['h_l_h']
    else :
        raw_l2 = None    
            
    # Calculate water level with air pressure adjustment (p = H rho g)                
    # Perform this only if p_wtr_l_cor-p_air_cor > p_dif_min and t_wtr_l_cor > t_wtr_min:    
    ds['h_l_h'] = calc_water_level(ds['p_wtr_l_cor'], 
                                   ds['p_air_cor'], 
                                   ds['h_l_dvr'])      
  
    # adding the raw water level data to the file
    if raw_l2 is not None:
        ds['h_l_h'] = ds['h_l_h'].fillna(raw_l2)
   
    # Checking if there is any water level data in the raw data
    if 'h_u_h' in ds:
        raw_l2 = ds['h_u_h']
    else :
        raw_l2 = None    
        
    # Perform this only if p_wtr_u_cor-p_air_cor > p_dif_min and t_wtr_u_cor > t_wtr_min:     
   
    ds['h_u_h'] = calc_water_level(ds['p_wtr_u_cor'], 
                                   ds['p_air_cor'],
                                   ds['h_u_dvr'])
    # adding the raw water level data to the file
    if raw_l2 is not None:
        ds['h_u_h'] = ds['h_u_h'].fillna(raw_l2)

    # Fill air temperature gaps with interpolated values
    print('Smoothing and interpolating atmospheric data...')
    ds['t_air_smooth'] = ds['t_air'].interpolate_na('time', method='linear').rolling(time=240).mean()
    ds['t_air_pos'] = ds['t_air_smooth'].where(ds['t_air_smooth']>0, other=0)
    ds['t_air_pos'] = ds['t_air_pos'].where(~np.isnan(ds['t_air_smooth']))

    print('L2 processing complete')
    return ds
    
def get_l3(L2):
    '''Perform L2 to L3 processing'''
    ds = L2.copy(deep=True)
  
    # Calculate diver discharge
    print('Deriving diver-only discharge...')
    ds['q_l_h'] = calc_discharge(ds['h_l_h'])
    # l2['q_l_h_unc'] = l2['q_l_h']*0.15 
    ds['q_u_h'] = calc_discharge(ds['h_u_h'])
    # l2['q_u_h_unc'] = l2['q_u_h']*0.15 
    ds['q_h'] = ds['q_u_h'].combine_first(ds['q_l_h'])    
    ds['q_h_unc'] = ds['q_h']*0.15     
    
    # Calculate diver + temperature discharge
    # Determined using IDL program Discharge_from_T_DMI  
    print('Deriving diver and temperature linked discharge...')
   
    ds['q_h_spring'] = 0.17*ds['t_air_pos']**3.4
    ds['q_h_autumn'] = 0.31*ds['t_air_pos']**3.4
    
    ds['q_h_spring'] = ds['q_h_spring'].where(ds['time.month']<=6)
    ds['q_h_autumn'] = ds['q_h_autumn'].where(ds['time.month']>=7)  
    
    ds['q_h_all'] = ds['q_h_spring'].combine_first(ds['q_h_autumn']) 
    ds['q_h_ext'] = ds['q_h'].combine_first(ds['q_h_all'])
    
    ds['q_h_ext_unc'] = ds['q_h_unc'].combine_first(ds['q_h_ext']*0.7)
    
    # Calculate cumulative discharge
    print('Deriving cumulative discharge...')
    # create an array of years (modify day/month for your use case)
    cum_group = xr.DataArray([t.year if ((t.month <= 4) or ((t.month==4) and (t.day < 1))) else (t.year + 1) for t in ds.indexes['time']], dims='time', name='my_years', coords={'time': ds['time']})
    
    # use that array of years (integers) to do the groupby
    ds['q_h_ext_cum'] = ds['q_h_ext'].groupby(cum_group).apply(lambda x: x.cumsum(dim='time', skipna=True))
    # l2['q_ext_h_cum'] = l2['q_ext_h'].cumsum(skipna=True)
    ds['q_h_ext_cum'] = ds['q_h_ext_cum'].where(ds['time.month'] > 4)
    ds['q_h_ext_cum'] = ds['q_h_ext_cum'].where(ds['time.month'] < 11)
    
    ds['q_h_ext_cum_unc'] = ds['q_h_ext_unc'].groupby(cum_group).apply(lambda x: x.cumsum(dim='time', skipna=True))    
    # l2['q_ext_h_cum_unc'] = l2['q_ext_h_unc'].cumsum(skipna=True)
    ds['q_h_ext_cum_unc'] = ds['q_h_ext_cum_unc'].where(ds['time.month'] > 4)
    ds['q_h_ext_cum_unc'] = ds['q_h_ext_cum_unc'].where(ds['time.month'] < 10)
    
    ds['q_h_ext_cum'] = ds['q_h_ext_cum']*1.e-9*3600. 
    ds['q_h_ext_cum_unc'] = ds['q_h_ext_cum_unc']*1.e-9*3600.     
    
    print('L3 processing complete')  
    return ds          
            
def get_config(config_file, inpath):
    '''Load configuration from .toml file. PROMICE .toml files support defining 
    features at the top level which apply to all nested properties, but do not 
    overwrite nested properties if they are defined
    
    Parameters
    ----------
    config_file : str
        TOML file path
    inpath : str
        Input folder directory where L0 files can be found
    
    Returns
    -------
    conf : dict
        Configuration dictionary
    '''
    conf = toml.load(config_file)                                              # Move all top level keys to nested properties,
    top = [_ for _ in conf.keys() if not type(conf[_]) is dict]                # if they are not already defined in the nested properties
    subs = [_ for _ in conf.keys() if type(conf[_]) is dict]                   # Insert the section name (config_file) as a file property and config file
    for s in subs:
        for t in top:
            if t not in conf[s].keys():
                conf[s][t] = conf[t]

        conf[s]['conf'] = config_file
        conf[s]['file'] = os.path.join(inpath, s)
        if not 'last_good_data_line' in conf[s]:
            conf[s]['last_good_data_line']=None

    for t in top: conf.pop(t)                                                  # Delete all top level keys beause each file
                                                                               # should carry all properties with it
    for k in conf.keys():                                                      # Check required fields are present
        for field in ["columns", "station_id", "format", "skiprows"]:
            assert(field in conf[k].keys())
    return conf

def load_l0(f, columns, skiprows, nodata, file_format, lastrow=None, 
            encode="ISO-8859-1"):
    '''Load Level 0 discharge data from file

    Parameters
    ----------
    f : str
        File path
    columns : list
        Ordered column names
    skiprows : int
        Number of rows to skip at beginning of file
    nodata : list/str/int
        No data value/s
    file_format : str
        File format
    lastrow : int, optional
        Last row of good data. The default is None.
    encode : str, optional
        File encoding. The default is "ISO-8859-1".

    Returns
    -------
    ds : xr.Dataset
        Dataset object
    '''
    if file_format.lower() == 'mon':
        s='       '
    elif file_format.lower() == 'csv':
        s=';'
    elif file_format.lower() == 'tx':
        s=','
    
    if lastrow==None:
        nr = None
    else:
        nr = lastrow-skiprows
        
    df = pd.read_csv(f, comment='#', sep=s, names=columns, skiprows=skiprows, 
                     na_values=nodata, index_col='time', skip_blank_lines=True, 
                     nrows=nr, encoding=encode, engine='python',
                     parse_dates=True)
    
    if not isinstance(df.index[0], pd._libs.tslibs.timestamps.Timestamp): 
        print('Inconsistent formatting. Attempting flexible loading...')
        df = load_l0_flexible(f, columns, skiprows, nodata, lastrow, encode)
    
    assert isinstance(df.index[0], pd._libs.tslibs.timestamps.Timestamp)
    
    # Drop SKIP columns
    for d in df.columns:
        if 'skip' in d.lower():
            df.drop(columns=d, inplace=True)
    
    # Carry to xarray.Dataset
    ds = xr.Dataset.from_dataframe(df)
    return ds
    
def load_l0_flexible(f, columns, skiprows, nodata, lastrow=None, encode="ISO-8859-1"):               
    '''Load Level 0 discharge data from file in a flexible manner that works
    regardless of file type and overrides inconsistent spacing between 
    variables. The downside is that this takes a bit longer than load_l0.

    Parameters
    ----------
    f : str
        File path
    columns : list
        Ordered column names
    skiprows : int
        Number of rows to skip at beginning of file
    nodata : list/str/int
        No data value/s
    lastrow : int, optional
        Last row of good data. The default is None.
    encode : str, optional
        File encoding. The default is "ISO-8859-1".

    Returns
    -------
    ds : xr.Dataset
        Dataset object
    '''
    new_cols = ['d','t'] 
    [new_cols.append(idx) for idx in columns[1:]]

    if lastrow==None:
        nr = None
    else:
        nr = lastrow-skiprows
    
    df = pd.read_csv(f, comment='#', delim_whitespace=True, names=new_cols, 
                     skiprows=skiprows, na_values=nodata, skip_blank_lines=True, 
                     nrows=nr, encoding=encode,engine='python') 
    time=[]
    [time.append(parserV1(dat, tim)) for dat,tim in zip(list(df['d']), list(df['t']))]
    df['time']=time
    df = df.set_index('time')
    df.drop(columns=['d','t'], inplace=True) 
    
    # Drop SKIP columns
    for d in df.columns:
        if 'skip' in d.lower():
            df.drop(columns=d, inplace=True)
    return df
              
def reformat_array(ds_arr):
    '''Reformat array of inconsistent entries to a numeric array'''
    a = ds_arr.attrs 
    ds_arr.values = pd.to_numeric(ds_arr, errors='coerce')
    ds_arr.attrs = a 
    return ds_arr 

def populate_cols(ds, names): 
    '''Populate array with empty variables based on a list of variable names'''      
    for v in names:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)      
    return ds 

def parserV1(d, t):
    '''Datetime parser for flexible Level 0 file loading'''                                      
    return pd.to_datetime(f'{d} {t}',
                          format='%Y/%m/%d %H:%M:%S.0')  

def offset_time(t, t_offset):
    '''Offset time in array
    
    Parameters
    ----------
    t : xr.DataArray
        Datetime array
    t_offset : int
        Datetime offset to apply (given in hours)
    
    Returns
    -------
    pd.DataFrame
        Datetime dataframe with applied offset
    '''
    df_t = t.to_dataframe()
    return  df_t['time'] + timedelta(hours=t_offset)

def calc_dh (dh_bolt, dh_diver, dH_0): 
    '''Calculate height of diver

    Parameters
    ----------
    dh_bolt : int
        Bolt position
    dh_diver : int
        Diver position
    dH_0 : int
        Height above reference, determined from stage-discharge relation

    Returns
    -------
    int
        Height of diver
    '''
    return dh_bolt - dh_diver - dH_0 
    
def offset_press(p, p_offset):
    '''Calculate pressure offset where pressure entries are not zero
    
    Parameters
    ----------
    p : xr.DataArray
        Pressure data
    p_offset : int
        Pressure reading offset
    
    Returns
    -------
    p : xr.DataArray
        Pressure data corrected for offset
    '''
    p = p.where(p != 0) 
    return p + p_offset

def resample_data(ds, t):
    '''Resample dataset to temporal averages
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to resample and average
    t : str
        Temporal resampling string indicator
    
    Returns
    -------
    xr.Dataset
        Resampled dataset
    '''
    df = ds.to_dataframe().resample(t).mean()
    vals = [xr.DataArray(data=df[c], dims=['time'], 
           coords={'time':df.index}, attrs=ds[c].attrs) for c in df.columns]
    return xr.Dataset(dict(zip(df.columns,vals)), attrs=ds.attrs)      

def calc_water_level(p_wtr, p_air, h_dvr, p_dif_min=5., t_wtr_min=-100.):
    '''Calculate water level with air pressure adjustment. Water temperature
    thresholding is currently not used in this calculation

    Parameters
    ----------
    p_wtr : xr.DataArray
        Pressure from diver
    p_air : xr.DataArray
        Air pressure
    h_dvr : xr.DataArray
        Height of diver
    p_dif_min : int, optional
        Minimum pressure difference (between p_wtr and p_air) to perform
        water level calculation. The default is 5.
    t_wtr_min : int, optional
        Minimum water temperature to perform water level calculation.
        The default is -100.

    Returns
    -------
    xr.DataArray
        Water level

    '''
    diff = p_wtr - p_air  
    diff = diff.where(diff > p_dif_min) 
    return h_dvr + (diff/98.2) 
       
def calc_discharge(H): # Version 3 by Dirk van As
    '''Calculate diver discharge. This is lifted directly from the IDL 
    processing scripts, version 3, by Dirk van As
    
    Parameters
    ----------
    H : xr.DataArray
        Water level
    
    Returns
    -------
    xr.DataArray
        Discharge array
    ''' 
    return 7.50536*H**2.34002 


def aws_plot(ds, v1,v2,st,threshold=2000):
    # Assuming you have an xarray dataset called 'ds' with variables 'var1', 'var2', and 'time'
    # Replace 'var1', 'var2', and 'time' with the actual names of your variables in the xarray dataset.
    # Get the current date
    current_date = pd.Timestamp.now()
    this_season = pd.Timestamp('2024-02-01')
    # Filter out future times
    ds = ds.where(ds['time'] <= current_date, drop=True)
    ds = ds.where(ds['time'] > this_season, drop=True)
    # Create a figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the first variable on the primary y-axis
    ax1.plot(ds['time'], ds[v1], label=f'{v1}', color='b')
    ax1.plot(ds['time'], (ds[v1]*0) +11.5, label=f'Batt_v Threshold', color='black')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(f'{v1}', color='b')  # Set y-axis label and color for var1
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(ds['time'], ds[v2], label=f'{v2}', color='r')
    ax2.set_ylabel(f'{v2}', color='r')  # Set y-axis label and color for var2
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add title
    plt.title(f'{v1} and {v2} at {st}')
    
    # Show plot
    plt.show()
if __name__ == "__main__":
    
    src_folder = os.getcwd()
    base_folder = os.path.abspath('..')
    
    l0_folder = base_folder + os.sep + 'level_0' + os.sep + 'tx'
    
    wat_b_f = l0_folder + os.sep + 'Watson10minTable.txt'
    wat_l_f = l0_folder + os.sep + 'GIOS_Watson_L_table.txt'
    rus_r_f = l0_folder + os.sep + 'GIOS_Russel_table.txt'
    lru_f = l0_folder + os.sep + 'GIOS_LR_table.txt'
    
    config_files = ['../config/watson_bridge_tx.toml','../config/GIOS_Watson_L.toml',\
                    '../config/GIOS_Russel.toml','../config/GIOS_LR.toml']

    tx_f = '../level_0/tx'
    
    st_ids = ['wat_b','wat_l','rus_r','lru']
    
    
    # Load config, variables CSF standards, and L0 files
    
    for cf,st in zip(config_files,st_ids):
        
       
        config = get_config(cf, tx_f)
        l0_list = get_l0(config)
        # Resample to hourly mean values
       
        ds_list=[]
        for i,ds in enumerate(l0_list):
            ds = resample_data(ds, '10min')
            ds_list.append(ds) 
        l1 = ds_list[0]
        for d in ds_list[1:]:
            l1 = l1.combine_first(d)
        aws_plot(l1,'batt_v','t_air',st)
        current_date = pd.Timestamp.now()
        this_season = pd.Timestamp('2024-02-01')
        # Filter out future times
 
        l1 = l1.where(ds['time'] < current_date, drop=True)
        l1 = l1.where(ds['time'] > this_season, drop=True)
        # Find the last valid (non-NaN) index for each variable in the dataset
        
        last_valid_index = l1['batt_v'].notnull()[::-1].argmax(dim='time')

        # Get the corresponding time value for this last valid index
        last_valid_time = l1['time'][::-1][last_valid_index]
        
        #int(last_valid_index) > 0:
        #    # Select data up to the last valid index, dropping NaNs after that point
        #    l1 = l1.sel(time=slice(l1['time'].min(), l1['time'][last_valid_index]))
            
            # Drop any remaining NaNs in the dataset along all dimensions
            #l1 = l1.dropna(dim='time', how='all')
            
            # Now ds_trimmed contains data up to the last non-NaN value
        print(f'Last Station Transmission at {st}')
        print(last_valid_time)
    
  
    



  
    
    
    