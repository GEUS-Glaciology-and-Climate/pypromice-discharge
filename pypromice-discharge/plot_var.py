# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:56:07 2024

@author: rabni
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

def read_csv(ff):
    return pd.read_csv(ff)

def read_nc(ff):
    return xr.open_dataset(ff)

def dvr_q_diff(ds_r,ds_t):
    
    q_l_raw = ds_r['q_l_h']  # q_h_ext er discharge som er blandet med air temp fra dmi og van essen heights, q_l_h er kun fra lower van essen
    q_u_raw = ds_r['q_u_h']
    
    q_l_tx = ds_t['q_l_h']
    q_u_tx = ds_t['q_u_h']       
    
    # Assuming your datasets are named ds1 and ds2
    common_time = pd.date_range(
        start=max(q_l_tx.time.min().values, q_l_raw.time.min().values),
        end=min(q_l_tx.time.max().values, q_l_raw.time.max().values),
        freq="H"  # hourly frequency
    )

    q_l_raw = q_l_raw.reindex(time=common_time)
    q_u_raw = q_u_raw.reindex(time=common_time)    
    q_l_tx = q_l_tx.reindex(time=common_time)
    q_u_tx = q_u_tx.reindex(time=common_time)    
    
    start_date = '2024-01-01'
    end_date = '2024-10-01'
    
    # l_analog_raw_mod = l_analog_raw.copy()
    # u_analog_raw_mod = u_analog_raw.copy()
    
    # l_dvr_tx_mod = l_dvr_tx.copy()
    # u_dvr_raw_mod = u_dvr_raw.copy()
    
    # value_to_subtract = 1.97
    # value_to_subtract = 0
    # value_to_subtract_dvr = 8
    
    # l_analog_raw_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract
    # u_analog_raw_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract
    
    # l_dvr_tx_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract_dvr
    # u_dvr_raw_mod.loc[dict(time=slice(start_date, end_date))] += value_to_subtract_dvr
    
    # diff_l = l_dvr_tx - l_analog_raw_mod
    # diff_u = u_dvr_raw_mod - u_analog_raw_mod
    
    
    return q_l_raw,q_u_raw,q_l_tx,q_u_tx


def dvr_h_diff(ds_r,ds_t):

    l_analog_raw = ds_r['h_l_h']  # q_h_ext er discharge som er blandet med air temp fra dmi og van essen heights, q_l_h er kun fra lower van essen
    u_analog_raw = ds_r['h_u_h']

    l_dvr_raw = ds_r['h_l_dvr']
    u_dvr_raw = ds_r['h_u_dvr']    
    
    l_dvr_tx = (ds_t['p_wtr_l'] * 0.01) #- 7.5
    u_dvr_tx = ds_t['p_wtr_u'] * 0.01       
    
    # Assuming your datasets are named ds1 and ds2
    common_time = pd.date_range(
        start=max(l_dvr_tx.time.min().values, l_analog_raw.time.min().values),
        end=min(l_dvr_tx.time.max().values, l_analog_raw.time.max().values),
        freq="H"  # hourly frequency
    )

    l_analog_raw = l_analog_raw.reindex(time=common_time)
    u_analog_raw = u_analog_raw.reindex(time=common_time)    
    l_dvr_tx = l_dvr_tx.reindex(time=common_time)
    u_dvr_tx = u_dvr_tx.reindex(time=common_time)    
    
    start_date = '2024-01-01'
    end_date = '2024-10-01'
    
    l_analog_raw_mod = l_analog_raw.copy()
    u_analog_raw_mod = u_analog_raw.copy()
    
    l_dvr_tx_mod = l_dvr_tx.copy()
    u_dvr_raw_mod = u_dvr_raw.copy()
    
    value_to_subtract = 1.97
    value_to_subtract = 0
    value_to_subtract_dvr = 0
    
    l_analog_raw_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract
    u_analog_raw_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract
    
    l_dvr_tx_mod.loc[dict(time=slice(start_date, end_date))] -= value_to_subtract_dvr
    u_dvr_raw_mod.loc[dict(time=slice(start_date, end_date))] += value_to_subtract_dvr
    
    diff_l = l_dvr_tx - l_analog_raw_mod
    diff_u = u_dvr_raw_mod - u_analog_raw_mod
    
    
    return diff_l,diff_u,l_analog_raw_mod,l_dvr_tx_mod

def plot_time_hist_two(v1,v2):
     
    # Create a figure with two subplots: one for the time series and one for the histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    day1 = '2023-04-01'
    day2 = '2024-10-01'
    v1 = v1.sel(time=slice(day1,day2))
    v2 = v2.sel(time=slice(day1,day2))
    # Plot the time series on the first subplot
    v1.plot.line(x='time', ax=ax1,c='red')  # Plotting time series
    v2.plot.line(x='time', ax=ax1,c='blue')  # Plotting time series
    ax1.set_title('Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    #plt.legend()
    v_arr = v1-v2
    # Plot the histogram on the second subplot
    v_arr.plot.line(x='time',ax=ax2,c='green')
    ax2.set_title('Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    return


def plot_time_hist(v_arr):
    
       
    # Create a figure with two subplots: one for the time series and one for the histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot the time series on the first subplot
    v_arr.plot.line(x='time', ax=ax1)  # Plotting time series
    ax1.set_title('Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    
    # Plot the histogram on the second subplot
    ax2.hist(v_arr.values.flatten()[~np.isnan(v_arr.values.flatten())], bins=100, color='skyblue', edgecolor='black')
    ax2.set_title('Histogram of Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    return

def plot_data_v(ds_r,ds_t):
    
    time_list_r = [pd.to_datetime(time).strftime('%Y-%m-%d-%H') for time in ds_r.time.values]
    time_list_t = [pd.to_datetime(time).strftime('%Y-%m-%d-%H') for time in ds_t.time.values]
    
    analog_raw = ds_r['h_l_h'].values.tolist()
    digital_tx = ds_t['h_l_dvr'].values.tolist()
    
    time_list_r = set([d[:10] for d,v in zip(time_list_r,analog_raw) if ~np.isnan(v)])
    time_list_t = set([d[:10] for d,v in zip(time_list_t,digital_tx) if ~np.isnan(v)])
    
    dates_l = [pd.to_datetime(list(time_list_r)),pd.to_datetime(list(time_list_t))]
    var = ['Analog Diver','Digital Diver']
    
    # date_range = pd.date_range(start='1982-01-01',end='2024-10-15', freq='D')
    # time_series_data = pd.Series(range(len(date_range)), index=date_range)
    
    #Plot the time series
    
    for i,dates in enumerate(dates_l):
        
        plt.figure(figsize=(20, 6))
        #plt.plot(time_series_data.index, time_series_data.values, label='Time Series Data')
        
        # Highlight specific dates with vertical lines
        for date in dates:
            plt.axvline(x=date, color='red', linestyle='-', label=f'Highlight {date.date()}')
      
        # Customizing the X axis ticks to show every year
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Tick at every year
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Format as 'YYYY'
        
        # Increase the size of the labels and titles
        plt.xticks(fontsize=12,rotation=90)  # X-axis label size
        plt.xlabel('Date', fontsize=14)  # X-axis title size
        plt.ylabel('Value', fontsize=14)  # Y-axis title size
        
        # Remove y-axis labels and ticks
        plt.gca().yaxis.set_ticks([])  # Remove ticks on the Y axis
        plt.gca().yaxis.set_ticklabels([])  # Remove labels on the Y axis
        plt.ylabel('')  # Optionally remove the Y-axis label title entirely
        # Optional: add title
        plt.title(f'{var[i]} Availabilty', fontsize=16)
        # Show legend and plot
        #plt.legend()
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    
    src_folder = os.getcwd()
    base_folder = os.path.abspath('..')
    
    l3_folder = base_folder + os.sep + 'level_3'
    
    w_b_raw_f = l3_folder + os.sep + 'watson_bridge_l3_raw.nc'
    w_b_tx_f = l3_folder + os.sep + 'watson_bridge_l3_tx.nc'
    w_b_tx_f_l1 = l3_folder + os.sep + 'watson_bridge_l1_tx.nc'
    
    ds_r = read_nc(w_b_raw_f)
    ds_t = read_nc(w_b_tx_f)
    ds_t_l1 = read_nc(w_b_tx_f_l1)
    
    diff_l,diff_u,raw_l,raw_dvr = dvr_h_diff(ds_r,ds_t_l1)
    q_l_raw,q_u_raw,q_l_tx,q_u_tx = dvr_q_diff(ds_r,ds_t)
    
    plot_time_hist( ds_r['t_air_pos'] )
    #plot_time_hist(diff_l)
    #plot_time_hist(raw_dvr)
    #plot_time_hist_two(q_l_raw,q_l_tx)
    #plot_time_hist_two(raw_l,raw_dvr)
    
    #plot_data_v(ds_r,ds_t)
    
    
    