# Watson River operational discharge processing

This repo is for the handling and processing of automated weather station (AWS) and discharge data, primarily for operational near-real-time processing of data from the Watson catchment monitoring programme.

AWS processing procedures are adopted from [pypromice](https://github.com/GEUS-Glaciology-and-Climate/pypromice), with specific station handling and discharge data processing developed here.

## Level 0 to Level 3 discharge workflow

`level_3` (`L3`) AWS and discharge data is produced from the original `L0 raw` data and station transmissions, `L0 tx`, with the following steps.

- `L0`: Original, untouched data

- `L1`: 
1. Arrays reformatted
2. Time offset applied
3. dH values calculated
4. Pressure offset applied
5. All data files combined into one

- `L2`:
1. Water level calculated
2. Air temperature interpolated and smoothed

- `L3`:
1. Diver-only discharge calculated
2. Temperature-and-diver discharge calculated
3. Cumulative discharge calculated 


### Data types

`L0` data can be provided in the following forms:

- `raw`
1. `.mon` barometer file format
2. `.csv` barometer file format
3. `.csv`/`.txt` file format, for the DMI air temperature files

- `tx`: `.txt` `L0tx` transmission file format, fetched using the [pypromice](https://github.com/GEUS-Glaciology-and-Climate/pypromice) `tx` module.


### Running the water module

`water.py` holds all the processing steps and workflow for producing `level_3` discharge data. It can be ran from a CLI as follows:

```
$ python3 water.py
```
