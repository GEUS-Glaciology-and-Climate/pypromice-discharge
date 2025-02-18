 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:37:47 2023

@author: pho
"""
import toml, os
import numpy as np
import pandas as pd
import csv
import xarray as xr
from datetime import timedelta, datetime

from argparse import ArgumentParser


def parse_arguments_watson():
    parser = ArgumentParser(description="watson raw l0->l3 processor")            
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='path to config files')
    parser.add_argument('-d', '--data', default=None, type=str, required=True, help='Path to tranmissions')
    parser.add_argument('-dm', '--dmi', default=None, type=str, required=True, help='Path to dmi data')                                        
    parser.add_argument('-o', '--out', default=None, type=str, required=True, help='path to raw data')         
    args = parser.parse_args()
    return args

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
    
def write_txt(ds,outdir,config_dir):
    '''Write Dataset to .csv file'''
    
    time_s = ['hourly','l0']
    sample_time = [None,None,60*24,60*24*365]
    
    time = pd.to_datetime(ds['time'].values)
    
    ds = ds.assign(
        year=('time', time.year),
        month=('time', time.month),
        dom=('time', time.day),
        hour=('time', time.hour),
        doy=('time', time.dayofyear),
        doc=('time', (time - pd.Timestamp('2000-01-01')).days)
    )
    
    for ts,sa in zip(time_s,sample_time):
        if sa:
            ds = resample_data(ds, f'{sa}min')
        ds = ds.fillna(-9999)    
        ds = ds.apply(lambda x: x.round(2))
        meta = pd.read_csv(config_dir+os.sep+f'meta_txt_{ts}.csv',delimiter=';')
        data_names = list(meta['Pipeline_Name'])
        output_names = list(meta['Output_Name'])
        units = list(meta['units'])
        
        print(output_names)
        
        df = pd.DataFrame({o:ds[d] for o,d in zip(output_names,data_names)})        
      
        df.columns = [col.split('_')[0] for col in df.columns]  # Remove any leading/trailing spaces
        df.columns = [o + ' ' + u if isinstance(u, str) else o for o,u in zip(df.columns,units)]
        df.columns = [col.strip() for col in df.columns]  # Remove any leading/trailing spaces

        df.to_csv(outdir+os.sep+f'Watson River Discharge (2006-2024) {ts}.txt' 
                  ,index=False,  # Do not include the index
                  encoding='utf-8',  # Specify UTF-8 encoding
                  header=True,  # Ensure header is included
                  sep='\t',      # Ensure consistent delimiter)
                  quoting=csv.QUOTE_NONE,  # Disable quoting
                  escapechar='\\'  # Escape special characters, if needed
                   )
   


def write_netcdf(ds, outfile):
    '''Write Dataset to .nc file'''
    ds.to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)             
    ds.close()
    
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
                if 'h_wtr_1' in ds:
                    ds['h_wtr_1'] = ds['h_l_h'] * 10**-2 # Going from centimeters water level to meters
                elif 'h_wtr_1' in ds:
                    ds['h_wtr_1'] = ds['h_u_h'] * 10**-2 # Going from centimeters water level to meters
            
            # Calculate upper and lower dH
            
            if 'dh_bolt_1' in c and 'dh_diver_1' in c:            
                h_dvr_1 = calc_dh(c['dh_bolt_1'], c['dh_diver_1'], c['dH_0']) 
                ds['h_dvr_1'] = (('time'), [h_dvr_1]*len(ds['time'].values))     
                
            if 'dh_bolt_2' in c and 'dh_diver_2' in c:
                h_dvr_2 = calc_dh(c['dh_bolt_2'], c['dh_diver_2'], c['dH_0']) 
                ds['h_dvr_2'] = (('time'), [h_dvr_2]*len(ds['time'].values))
            
            if 'dh_bolt_3' in c and 'dh_diver_3' in c:            
                h_dvr_3 = calc_dh(c['dh_bolt_3'], c['dh_diver_3'], c['dH_0']) 
                ds['h_dvr_3'] = (('time'), [h_dvr_3]*len(ds['time'].values))   
                
                if 'dh_bolt_2' not in c and 'dh_diver_2' not in c:
                    h_dvr_2 = calc_dh(c['dh_bolt_3'], c['dh_diver_3'], c['dH_0']) 
                    ds['h_dvr_2'] = (('time'), [h_dvr_2]*len(ds['time'].values))
                
            # Mask out where p_wtr_u and p_wtr_l are nan to also be nan here
            
            # Apply pressure offset, also apply to p_wtr_l, p_air
            if hasattr(ds, 'p_wtr_1'):
                ds['p_wtr_1_cor'] = offset_press(ds['p_wtr_1'], c['p_offset_1'])
            if hasattr(ds, 'p_wtr_2'): 
                ds['p_wtr_2_cor'] = offset_press(ds['p_wtr_2'], c['p_offset_2']) 
                ds['p_wtr_2_cor'] = ds['p_wtr_2_cor'].where(ds['p_wtr_2_cor'] != 1450.0)
            if hasattr(ds, 'p_wtr_3'): 
                ds['p_wtr_3_cor'] = offset_press(ds['p_wtr_3'], c['p_offset_3']) 
                ds['p_wtr_3_cor'] = ds['p_wtr_3_cor'].where(ds['p_wtr_3_cor'] != 1450.0)
            if hasattr(ds, 'p_air_baro'):
                ds['p_air_baro_cor'] = offset_press(ds['p_air_baro'], c['p_offset_a'])
  
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
        
        l1 = xr.merge([l1,l0_air],compat='override')
        if hasattr(ds, 't_air_baro'):
            l1['t_air_comb'] = xr.merge([l1['t_air_baro'], l0_air['t_air_dmi']],compat='override')
        else:
            l1['t_air_comb'] =  l1['t_air_dmi'] 
        #l1 = xr.merge([l1,l0_air],compat='override')
        
        
    
    print('L1 processing complete')
    return l1
        
def get_l2(L1):
    '''Perform L1 to L2 processing'''
    ds = L1.copy(deep=True)   
    print('Calculating water level...')
        
    # Checking if there is any water level data in the raw data
    if 'h_wtr_1' in ds:
        raw_l2 = ds['h_wtr_1']
    else :
        raw_l2 = None    
            
    # Calculate water level with air pressure adjustment (p = H rho g)                
    # Perform this only if p_wtr_l_cor-p_air_cor > p_dif_min and t_wtr_l_cor > t_wtr_min:    
    ds['h_wtr_1'] = calc_water_level(ds['p_wtr_1_cor'], 
                                   ds['p_air_baro_cor'], 
                                   ds['h_dvr_1'])      
  
    # adding the raw water level data to the file
    if raw_l2 is not None:
        ds['h_wtr_1'] = ds['h_wtr_1'].fillna(raw_l2)
   
    # Checking if there is any water level data in the raw data
    if 'h_wtr_2' in ds:
        raw_l2 = ds['h_wtr_2']
    else :
        raw_l2 = None
        
    # Perform this only if p_wtr_u_cor-p_air_cor > p_dif_min and t_wtr_u_cor > t_wtr_min:
   
    ds['h_wtr_2'] = calc_water_level(ds['p_wtr_2_cor'], 
                                   ds['p_air_baro_cor'],
                                   ds['h_dvr_2'])
    # adding the raw water level data to the file
    if raw_l2 is not None:
        ds['h_wtr_2'] = ds['h_wtr_2'].fillna(raw_l2)
    
    
    ds['h_wtr_3'] = calc_water_level(ds['p_wtr_3_cor'],
                                   ds['p_air_baro_cor'],
                                   ds['h_dvr_3'])

    # Fill air temperature gaps with interpolated values
    
    # Fill air temperature gaps with interpolated values
    print('Smoothing and interpolating atmospheric data...')
    ds['t_air_interp'] = ds['t_air_comb'].interpolate_na('time', method='linear')
    ds['t_air_smooth'] = ds['t_air_interp'].rolling(time=240).mean()
    
    ds['t_air_pos'] = ds['t_air_smooth'].where(ds['t_air_smooth']>0, other=0)
    ds['t_air_pos'] = ds['t_air_pos'].where(~np.isnan(ds['t_air_smooth']))

    print('L2 processing complete')
    return ds
    
def get_l3(L2):
    '''Perform L2 to L3 processing'''
    ds = L2.copy(deep=True)
  
    # Calculate diver discharge
    print('Deriving diver-only discharge...')
    ds['q_wtr_1'] = calc_discharge(ds['h_wtr_1'])
    # l2['q_l_h_unc'] = l2['q_l_h']*0.15 
    ds['q_wtr_2'] = calc_discharge(ds['h_wtr_2'])
    # l2['q_u_h_unc'] = l2['q_u_h']*0.15 
    ds['q_wtr_3'] = calc_discharge(ds['h_wtr_3'])
    
    
    ds['h_wtr_comb'] = ds['h_wtr_1'].combine_first(ds['h_wtr_2']).combine_first(ds['h_wtr_3'])   
    ds['t_wtr_comb'] = ds['t_wtr_1'].combine_first(ds['t_wtr_2']).combine_first(ds['t_wtr_3'])   
    
    
    ds['q_wtr_comb'] = ds['q_wtr_1'].combine_first(ds['q_wtr_2']).combine_first(ds['q_wtr_3'])   
    ds['q_wtr_comb_unc'] = ds['q_wtr_comb']*0.15
    
    # Calculate diver + temperature discharge
    # Determined using IDL program Discharge_from_T_DMI  
    print('Deriving diver and temperature linked discharge...')
   
    ds['q_wtr_mod_spring'] = 0.17*ds['t_air_pos']**3.4
    ds['q_wtr_mod_autumn'] = 0.31*ds['t_air_pos']**3.4
    
    ds['q_wtr_mod_spring'] = ds['q_wtr_mod_spring'].where(ds['time.month']<=6)
    ds['q_wtr_mod_autumn'] = ds['q_wtr_mod_autumn'].where(ds['time.month']>=7)  
    
    ds['q_wtr_mod_all'] = ds['q_wtr_mod_spring'].combine_first(ds['q_wtr_mod_autumn']) 
    ds['q_wtr_ext'] = ds['q_wtr_comb'].combine_first(ds['q_wtr_mod_all'])
    
    ds['q_wtr_ext_unc'] = ds['q_wtr_comb_unc'].combine_first(ds['q_wtr_ext']*0.7)
    
    # Calculate cumulative discharge
    print('Deriving cumulative discharge...')
    # create an array of years (modify day/month for your use case)
    cum_group = xr.DataArray([t.year if ((t.month <= 4) or ((t.month==4) and (t.day < 1))) else (t.year + 1) for t in ds.indexes['time']], dims='time', name='my_years', coords={'time': ds['time']})
    
    # use that array of years (integers) to do the groupby
    ds['q_wtr_ext_cum'] = ds['q_wtr_ext'].groupby(cum_group).apply(lambda x: x.cumsum(dim='time', skipna=True))
    # l2['q_ext_h_cum'] = l2['q_ext_h'].cumsum(skipna=True)
    ds['q_wtr_ext_cum'] = ds['q_wtr_ext_cum'].where(ds['time.month'] > 4)
    ds['q_wtr_ext_cum'] = ds['q_wtr_ext_cum'].where(ds['time.month'] < 11)
    
    ds['q_wtr_ext_cum_unc'] = ds['q_wtr_ext_unc'].groupby(cum_group).apply(lambda x: x.cumsum(dim='time', skipna=True))    
    # l2['q_ext_h_cum_unc'] = l2['q_ext_h_unc'].cumsum(skipna=True)
    ds['q_wtr_ext_cum_unc'] = ds['q_wtr_ext_cum_unc'].where(ds['time.month'] > 4)
    ds['q_wtr_ext_cum_unc'] = ds['q_wtr_ext_cum_unc'].where(ds['time.month'] < 10)
    
    ds['q_wtr_ext_cum'] = ds['q_wtr_ext_cum']*1.e-9*3600. 
    ds['q_wtr_ext_cum_unc'] = ds['q_wtr_ext_cum_unc']*1.e-9*3600.     
    
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

if __name__ == "__main__":
    
    args = parse_arguments_watson()
     
    config_dir = args.config
    dmi_dir = args.dmi
    l0_dir = args.data
    out_dir = args.out
    air_config_file = config_dir + os.sep + 'dmi_air.toml'
    
    meta = pd.read_csv(config_dir + os.sep + 'station_meta.csv')
    tx_name = meta['raw_name']
    st_name = meta['out_raw_name']
    
    # Bridge station site raw data processing
    print('Commencing station tx processing...')
    
    for tx,st in zip(tx_name,st_name):
        out = out_dir + os.sep + st
        config_file = config_dir + os.sep + f'{tx}.toml'
        ds = process(l0_dir, config_file, air_config_file, dmi_dir)
        write_csv(ds, f'{out}.csv')
        write_txt(ds, f'{out}.txt',config_dir)
        write_netcdf(ds, f'{out}.nc')
    
    
    
    # Uploading to Dataverse
    
    
    

    
    