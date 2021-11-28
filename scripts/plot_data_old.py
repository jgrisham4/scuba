#!/usr/bin/env python

'''
Simple script for plotting scuba dive data.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import collections
import datetime

DataPoint = collections.namedtuple('DataPoint', ['number', 'date', 'time',
                                                 'elapsed_time', 'depth',
                                                 'temperature', 'pressure'])
Dive = collections.namedtuple('Dive', ['number', 'date', 'time', 'elapsed_time',
                                       'depth', 'temperature', 'pressure',
                                       'tank_volume'])

def import_dives(dive_file_name, tank_file_name):
    '''Imports dive data.

    @param dive_file_name name of dive file to import.
    @param tank_file_name name of tank volume file to import.
    @return a list of Dive namedtuples.
    '''

    # Make sure file exists.
    if not os.path.isfile(dive_file_name):
        raise ValueError(f'File {dive_file_name} does not exist.')

    # Read file contents.
    with open(dive_file_name) as dive_file:
        lines = dive_file.readlines()[1:]

    # Remove double quotes.
    no_quote_lines = list(map(lambda l: l.replace('"', ''), lines))

    # Parse each line.
    dive_numbers = []
    dates = []
    times = []
    dive_times = []
    depths = []
    temperatures = []
    pressures = []
    for line in no_quote_lines:
        clean_line = list(map(lambda f: f.strip(), line.split(',')))
        dive_numbers.append(int(clean_line[0]))
        dates.append(clean_line[1])
        times.append(clean_line[2])
        dive_times.append(clean_line[3])
        depths.append(float(clean_line[4]))
        if not clean_line[5]:
            temperatures.append(temperatures[-1])
        else:
            temperatures.append(float(clean_line[5]))
        pressures.append(float(clean_line[6]))

    # Compute number of seconds from elapsed dive times.
    dive_times_sec = []
    for time in dive_times:
        split_time = time.split(':')
        minutes = float(split_time[0])
        seconds = float(split_time[1])
        dive_times_sec.append(60.0 * minutes + seconds)

    # Instantiate data points.
    results = zip(dive_numbers, dates, times, dive_times_sec,
                  depths, temperatures, pressures)
    datapoints = []
    for num, date, time, dive_time, depth, temp, press in results:
        datapoints.append(DataPoint(num, date, time, dive_time, depth, temp,
                                    press))

    # Find all unique dives.
    unique_dive_numbers = list(set(dive_numbers))
    print(f'Found {len(unique_dive_numbers)} dives.')

    # Import tank information.
    if not os.path.isfile(tank_file_name):
        raise ValueError('File {tank_file_name} does not exist.')
    tank_info = pd.DataFrame.from_csv()

    # Instantiate Dive objects.
    for dive_number in unique_dive_numbers:

Dive = collections.namedtuple('Dive', ['number', 'date', 'time', 'elapsed_time',
                                       'depth', 'temperature', 'pressure',
                                       'tank_volume'])


    return []

# Import dive data.
dive_data = import_dives('dive_profiles.csv')

# Kind of need to include tank size in this data.  That way, you can compute mass of gas that you used.

# Use ideal gas law for now.
