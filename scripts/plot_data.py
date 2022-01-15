'''
This script imports and analyzes scuba dive data.

@author James Grisham
@date 10/13/2021
'''

import os
import re
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cantera as ct

P_SURFACE = 14.7 * 144.0 # lbf/ft2
GRAV = 32.174            # ft/s2
RHO_WATER = 1.940        # slug/ft3
GAS_CONSTANT = 1716.49   # ft-lbf/slug-R

# Define conversion factors from imperial to base units.
C_PSI = 144.0

# Define conversion factors from metric to imperial base units.
C_KG = 1.0               # FIXME
C_M = 1.0                # FIXME
C_PA = 1.0               # FIXME
C_KELVIN = 1.0           # FIXME

def cleanup_columns(orig_df):
    '''This function cleans up column names of pandas dataframes.

    This function replaces whitespaces with underscores and units in
    parenthesis with a double underscore.
    '''

    # Get old names and remove extraneous whitespace.
    orig_names = orig_df.columns
    old_names = [s.strip() for s in orig_names]

    # Find units.
    find_units = lambda s: re.search(r'\((\w+)\)', s)
    matches = list(map(find_units, old_names))
    units = []
    for match in matches:
        if match:
            units.append(match.group(1))
        else:
            units.append(None)

    # Remove units from column name.
    remove_units = lambda s: re.sub(r'\s*\(\w*\)', '', s)
    names_no_units = list(map(remove_units, old_names))

    # Replace spaces with underscores.
    clean_names = list(map(lambda s: re.sub(' ', '_', s).lower(),
                           names_no_units))

    # Add units back in with two underscores.
    new_names = []
    for name, unit in zip(clean_names, units):
        if unit is not None:
            new_names.append(f'{name}__{unit}')
        else:
            new_names.append(name)

    # Rename columns.
    rename_dict = dict(zip(orig_names, new_names))
    new_df = orig_df.rename(columns=rename_dict)

    return new_df

def separate_dives(dive_info):
    '''Separates pandas DataFrame into list of dataframes for each dive.

    @param dive_info pandas.DataFrame which contains dive profiles.
    @return a list of dataframes, one for each dive.
    '''

    unique_dive_ids = list(set(dive_info.dive_number))
    print(f'Found {len(unique_dive_ids)} unique dives.')
    tables = []
    for unique_dive_id in unique_dive_ids:
        tables.append(dive_info[dive_info.dive_number == unique_dive_id])
    return tables

def convert_to_sec(min_str):
    '''Converts mm:ss string to a float representation in seconds.

    @param min_str minute string from CSV file format ("mm:ss").
    @return a float representation in seconds.
    '''

    split_str = min_str.split(':')
    return float(split_str[0]) * 60.0 + float(split_str[1])

def ideal_gas_density(pressure, temperature):
    '''Simple function which uses the ideal gas law (assuming air) to compute density.

    @param pressure pressure in psi.
    @param temperature temperature in deg F.
    @return density in slug/ft3.
    '''

    density = pressure / (GAS_CONSTANT * temperature)
    return density

def real_gas_density(pressure: float, temperature: float, composition: Dict[str, float]) -> float:
    '''Uses cantera to compute density.
    '''

    gas = ct.Solution('air_below6000K.cti')
    gas.X = composition
    gas.TP = pressure * C_PA, temperature * C_KELVIN
    gas.equilibrate('TP') # TODO Make sure this makes sense...

    return gas.density / (C_KG / C_M**3)

def compute_mass(df_in):
    '''Computes mass as a function of time.

    rho = f(p,T)

    m = rho V
    '''

    # Fill in temperature data.
    temps = df_in.sample_temperature__F.values
    last_temp = temps[0]
    for i, temp in enumerate(temps):
        if np.isnan(temp):
            temps[i] = last_temp
        else:
            last_temp = temp

    # Fill in temperature in Rankine.
    df_in['temperature__R'] = df_in.sample_temperature__F + 459.67

    # Compute density.
    pressures = df_in.sample_pressure__psi.values * 144.0 # psf
    temperatures = df_in.temperature__R.values
    composition = {'O2': df_in.o2.values[0]/100.0, 'N2': df_in.n2.values[0]/100.0}
    #densities = ideal_gas_density(pressures, temperatures)
    density_helper = lambda p, t: real_gas_density(p, t, composition)
    densities = list(map(density_helper, pressures, temperatures))
    df_in['density__slugqft3'] = densities

    # Compute mass of breathing gas.
    mass = densities * df_in.tank_volume__ft3
    df_in['mass__slug'] = mass

    return df_in

def compute_volumetric_consumption(df_in):
    '''Function for computing volumetric gas consumption.
    '''

    # Make sure mass has already been computed.
    if 'mass__slug' not in df_in.columns:
        df_in = compute_mass(df_in)

    # Compute mdot.
    time = df_in.elapsed_time__sec.values
    mass = df_in.mass__slug.values
    mdot = -np.gradient(mass) / np.gradient(time)
    df_in['mdot__slugqs'] = mdot

    # Compute volumetric flowrate.
    df_in['volumetric_flow_rate__ft3qs'] = mdot / df_in.density__slugqft3.values

    # SAC = dP / (dt * p(d_avg))
    surface_avg_consumption = compute_sac(df_in)

    return df_in, surface_avg_consumption

def compute_sac(df_in):
    '''Computes surface air consumption.

    SAC = dP / (dt * p(d_avg))
    '''

    time = df_in.elapsed_time__sec.values
    tank_pressure = df_in.sample_pressure__psi.values * C_PSI
    dive_duration = time[-1] - time[0]
    average_depth = np.mean(df_in.sample_depth__ft)
    avg_water_pressure = P_SURFACE + RHO_WATER * GRAV * average_depth
    surface_air_consumption = -(tank_pressure[-1] - tank_pressure[0]) / \
            (dive_duration * avg_water_pressure)

    return surface_air_consumption

def compute_rmv(df_in, sac):
    '''Computes respiratory minute volume.

    RMV = SAC * tank_volume / p_operating
    '''

    return []


# Read dive computer data, cleanup names, and delete unnecessary columns.
proj_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(proj_path, 'data')
dive_profile_path = os.path.join(data_dir, 'dive_profiles.csv')
dive_profiles = cleanup_columns(pd.read_csv(dive_profile_path))
del_cols = ['sample_heartrate']
dive_profiles = dive_profiles.drop(del_cols, axis=1)

# Read tank info.
tank_data_path = os.path.join(data_dir, 'tank_info.csv')
tank_data = cleanup_columns(pd.read_csv(tank_data_path))

# Populate tank info.
full_table = pd.DataFrame.merge(dive_profiles, tank_data, left_on='dive_number',
                                right_on='dive_number', how='left')

# Compute elapsed time in seconds.
min_strs = list(full_table.sample_time__min.values)
full_table['elapsed_time__sec'] = np.array(list(map(convert_to_sec, min_strs)))

# Separate tables per dive.
sep_profiles = separate_dives(full_table)

# Plot mass vs time.
fig2 = plt.figure()
sep_profiles_mass_sac = list(map(compute_volumetric_consumption, sep_profiles))
sep_profiles_mass = [output[0] for output in sep_profiles_mass_sac]
sac = [output[1] for output in sep_profiles_mass_sac]
for profile in sep_profiles_mass:
    dive_id = list(set(profile.dive_number))
    label = f'Dive {dive_id[0]}'
    plt.plot(profile.elapsed_time__sec/60.0,
             profile.mass__slug, label=label)
plt.xlabel('Time (min)')
plt.ylabel('Gas mass (lbm)')
plt.grid()
plt.legend()

# Plot volumetric flow rate vs time.
fig3 = plt.figure()
for profile in sep_profiles_mass:
    dive_id = list(set(profile.dive_number))
    label = f'Dive {dive_id[0]}'
    plt.plot(profile.elapsed_time__sec/60.0,
             profile.volumetric_flow_rate__ft3qs,
             label=label)
plt.xlabel('Time (min)')
plt.ylabel('Volumetric consumption (ft$^3$/min)')
plt.grid()
plt.legend()

# Plot mean volumetric consumption vs SAC.
fig4 = plt.figure()
for profile, sac in sep_profiles_mass_sac:
    dive_id = list(set(profile.dive_number))
    plt.plot(dive_id, sac ,'sk')
    plt.plot(dive_id, np.mean(profile.volumetric_flow_rate__ft3qs.values)*60.0, 'ob')
plt.xlabel('Time (min)')
plt.ylabel('Mean volumetric consumption (ft$^3$/min)')
plt.grid()

plt.show()
