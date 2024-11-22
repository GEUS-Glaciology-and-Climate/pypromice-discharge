#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:30:18 2023

Script to reformat metadata_divers.csv to config .toml file for pypromice
compatibility

@author: pho
"""

def format_cols(m, header):
    col = '["time",'
    count=1
    for i in list(range(0,10))[7:]:
        flag = False
        for j in list(range(len(m)))[14:]:
            if i == int(m[j]):
                idx = j 
                flag = True
        if flag==True:
            name = header[idx].split('col_')[-1].lower()
            col=col+'"' + name + '"' +','
        # else:
        #     col=col+f'"SKIP_{str(count)}",'
        
        count=count+1
        
    col=col[:-1]+']'
    return col
    

with open('metadata_divers.csv', 'r') as f:
    lines = f.readlines()
    header = lines[0].split(',')
    meta = lines[1:]


with open('watson_discharge.toml', 'w') as f:
    f.write('station_id = "watson_discharge"\ndH_0=9.01\nnodata=["-999", "NAN"]\n\n\n')
    
    for line in meta:
        m = line.split(',')
        f.write(f'["{m[0]}"]\n')
        f.write(f'format="{m[0].split(".")[-1].lower()}"\n')
        f.write(f'skiprows={int(m[2])-1}\n')
        f.write(f'last_good_data_line={int(m[3])+1}\n')
        f.write(f'utc_offset={m[5]}\n')
        f.write(f'dh_bolt={m[6]}\n')
        f.write(f'dh_diver={m[7]}\n')
        f.write(f'p_offset={m[8]}\n')
        cols = format_cols(m, header)
        f.write(f'columns={cols}\n\n')        