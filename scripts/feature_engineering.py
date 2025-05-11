# %%
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import IPython.display as display
import tensorflow as tf

# %%
def drop_before(dataframe, year):
    return dataframe[dataframe['Time'].dt.year >= year]

# %%
def apply_lag(dframe, auto_clean=False):
    result = dframe.groupby('StormID').apply(lambda group: group.assign(
        LatLag1=group['Lat'].shift(1),
        LonLag1=group['Lon'].shift(1)
    ))

    if auto_clean:
        result = clean_lag(result)

    return result

def clean_lag(dframe):
    return dframe.dropna()

# %%
def append_unit_sphere_coord(dframe, with_lag=False):
    dframe['x'] = np.cos(np.radians(dframe.Lat)) * np.cos(np.radians(dframe.Lon))
    dframe['y'] = np.cos(np.radians(dframe.Lat)) * np.sin(np.radians(dframe.Lon))
    dframe['z'] = np.sin(np.radians(dframe.Lat))
    
    if with_lag:
        dframe['xLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.cos(np.radians(dframe.LonLag1))
        dframe['yLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.sin(np.radians(dframe.LonLag1))
        dframe['zLag1'] = np.sin(np.radians(dframe.LatLag1))

    return dframe

# %%
import pyproj

def append_lambert_conformal_conic(dframe, with_lag=False):
    # This is the lambert conformal conic projection
    # The projection is in the form of x, y, z
    # The result is appended to the dataframe with keys 'LambertX', 'LambertY', 'LambertZ'
    # The result is appended to the dataframe with keys 'LambertXLag1', 'LambertYLag1', 'LambertZLag1'
    # if with_lag is True
    # The function returns the dataframe
    # The function raises ValueError if the keys 'Lat' and 'Lon' are not found in the dataframe
    if(('Lat' not in dframe.keys()) or ('Lon' not in dframe.keys())):
        raise ValueError('Lat and Lon not found in dataframe')

    # The lambert conformal conic projection
    # The projection is in the form of x, y, z
    dframe['LambertX'], dframe['LambertY'] = zip(*dframe.apply(lambda row: convert_from_lon_lat_to_lambert(row['Lon'], row['Lat']), axis=1))


    if with_lag:
        if(('LatLag1' not in dframe.keys()) or ('LonLag1' not in dframe.keys())):
            raise ValueError('Lagged coordinates not found in dataframe')

        dframe['LambertXLag1'], dframe['LambertYLag1'] = zip(*dframe.apply(lambda row: convert_from_lon_lat_to_lambert(row['LonLag1'], row['LatLag1']), axis=1))

    return dframe

def convert_from_lon_lat_to_lambert(lon, lat):
    proj_daymet = "+proj=lcc +lat_1=0 +lat_2=60 +lon_0=140 +lon_1=100 +lon_2=280 +datum=WGS84 +no_defs" #my custom CRS
    return pyproj.Proj(proj_daymet)(lon, lat)

# %%
# This function performs the lambert canonical confirmal porjection using the lat and lon
# def append_lambert_conformal_canonical(dframe, with_lag=False):
# %%

def translational_direction(lon, lat, lonlag1, latlag1):
    return tf.math.atan2(lat - latlag1, lon - lonlag1)

# Calculates the translation direction
# The result is appended to the dataframe with key 'TransDir'
def append_translational_direction(dframe):
    if(('LatLag1' not in dframe.keys()) or ('LonLag1' not in dframe.keys())):
        raise ValueError('Lagged coordinates not found in dataframe')
    
    dframe['TransDir'] = dframe.apply(lambda row: (translational_direction(row['Lon'], row['Lat'], row['LonLag1'], row['LatLag1'])).numpy(), axis=1)
    # delta_lat = dframe['Lat'] - dframe['LatLag1']
    # delta_lon = dframe['Lon'] - dframe['LonLag1']
    # dframe['TransDir'] = np.arctan2(delta_lat, delta_lon)

# %% This is used to calculate the haversine distance given the lon and lat of two points

# This is used to calculate the haversine distance given the lon and lat of two points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    # lon1, lat1, lon2, lat2 = map(tfm, [lon1, lat1, lon2, lat2])

    def radians(degrees):
        return tf.math.multiply(degrees, tf.constant(np.pi / 180.0, dtype=tf.float32))

    # lon1, lat1, lon2, lat2 = tf.map_fn(radians, [lon1, lat1, lon2, lat2])
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)

    # haversine formula 
    # dlon = lon2 - lon1 
    # dlat = lat2 - lat1 
    # a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    # c = 2 * arcsin(sqrt(a)) 
    # r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    # return c * r
    dlon = tf.math.subtract(lon2, lon1)
    dlat = tf.math.subtract(lat2, lat1)
    a = tf.math.add(
        tf.square(tf.sin(tf.math.divide_no_nan(dlat, 2))),
        tf.multiply(
            tf.multiply(tf.cos(lat1), tf.cos(lat2)),
            tf.square(tf.sin(tf.math.divide_no_nan(dlon, 2)))
        )
    )
    c = tf.multiply(2, tf.asin(tf.sqrt(a)))
    r = tf.constant(6371.0, dtype=tf.float32)
    return tf.multiply(c, r)
    

# %%
def tranlsational_spped(lon, lat, lonlag1, latlag1, time=3):
    # return  haversine(lonlag1, latlag1, lon, lat) / time
    return tf.math.divide_no_nan(haversine(lonlag1, latlag1, lon, lat), time)

# Calculates the translation speed
# The result is appended to the dataframe with key 'TransSpeed'
def append_translational_speed(dframe, time=3):
    if(('LatLag1' not in dframe.keys()) or ('LonLag1' not in dframe.keys())):
        raise ValueError('Time or lagged coordinates not found in dataframe')

    # dframe['TransSpeed'] = haversine(dframe['LonLag1'], dframe['LatLag1'], dframe['Lon'], dframe['Lat']) / time
    dframe['TransSpeed'] = dframe.apply(lambda row: tranlsational_spped(row['Lon'], row['Lat'], row['LonLag1'], row['LatLag1'], time).numpy(), axis=1)

# %%
# This function will filter out the number of points that are not in the three hour time grame
def drop_in_between_observation(dframe, hour=3):
    return dframe[dframe['Time'].dt.hour % 3 == 0]

# %%
# This function will remove the groups where the number of observations is less than 10
def drop_less_than_n_observation(dframe, n=10):
    return dframe.groupby('StormID').filter(lambda x: len(x) >= n)

# %%
def check_time_series(dframe, use_wind=True):
    problematic_storms_id = []
    problematic_storms_id_debug_info = []
    for name, val in dframe.groupby('StormID'):
        for index in range(1, len(val)):
            if val['Time'].iloc[index] - val['Time'].iloc[index - 1] != pd.Timedelta('3 hours'):
                if not use_wind:
                    problematic_storms_id.append(name)
                    problematic_storms_id_debug_info.append((name, val['Time'].iloc[index], val['Time'].iloc[index - 1]))
                elif val['Time'].iloc[index] - val['Time'].iloc[index - 1] != pd.Timedelta('6 hours'):
                    problematic_storms_id.append(name)
                    problematic_storms_id_debug_info.append((name, val['Time'].iloc[index], val['Time'].iloc[index - 1]))
    return problematic_storms_id, problematic_storms_id_debug_info

# %%
def create_features(use_wind=True):
    # %%
    xrds = xr.open_dataset('../data/raw/IBTrACS.WP.v04r01.nc')
    # dimensions = xrds.dims
    # coords = xrds.coords
    # display.display(xrds)
    # use_wind = False

    # %%
    features_selected = ['sid', 'time', 'lat', 'lon']

    if use_wind:
        features_selected += ['wmo_wind', 'wmo_pres', 'wmo_agency']
    xrds_selected = xrds[features_selected]
    df = xrds_selected.to_dataframe().reset_index()
    df = df[features_selected]

    df.rename(columns={'sid': 'StormID', 'time': 'Time', 'lat': 'Lat', 'lon': 'Lon'}, inplace=True)
    # print(df.head())

    # %%
    df_post_1900 = drop_before(df, 1900)
    df_post_1900.reset_index(drop=True, inplace=True)
    # display.display(df_post_1900)

    # %%
    # print(df_with_wind.groupby('wmo_agency').size())
    time_step = 3
    if use_wind:
        df_with_wind = df_post_1900[~df_post_1900['wmo_wind'].isna()]
        print(f'The number before removing observations without wind speed: {len(df_post_1900)}')
        print(f'The number of observations loss due to including wind speed: {len(df_post_1900) - len(df_with_wind)}')
        print(df_with_wind.groupby('wmo_agency').size())

        # List of values to remove
        values_to_remove = [b'atcf', b'hurdat_epa', b'newdelhi']

        # Filter out the rows where 'wmo_agency' matches any of the values in the list, but keep empty strings
        df_with_wind_filtered = df_with_wind[~df_with_wind['wmo_agency'].isin(values_to_remove) | (df_with_wind['wmo_agency'] == b'')]
        print(f'Remaing agencies: {df_with_wind.groupby("wmo_agency").size()}')
        print(f'Remaing total: {len(df_with_wind)}')

        # We also have to drop ones that are done by agencies using 1 minute or 3 minute winds
        # display.display(df_with_wind)
        df_post_1900 = df_with_wind
        time_step = 6

    # %%
    print(f'The number of observations before removing in between observations: {len(df_post_1900)}')
    df_post_1900_fixed_interval = drop_in_between_observation(df_post_1900, hour=3)
    print(f'The number of observations after removing in between observations: {len(df_post_1900_fixed_interval)}')
    print(f'Amount lost: {len(df_post_1900) - len(df_post_1900_fixed_interval)}')

    # %%
    # We have to check if the time series is correct
    df_post_1900_fixed_interval.reset_index(drop=True, inplace=True)
    # display.display(df_post_1900_fixed_interval)

    # %%
    problematic_storms_id, _ = check_time_series(df_post_1900_fixed_interval, use_wind=use_wind)
    # display.display(problematic_storms_id)
    print(len(problematic_storms_id))
    print(len(df_post_1900_fixed_interval.groupby('StormID').groups.keys()))

    # %%
    # I have noticed that there are many storms that fit into this category.
    # I think it is worthwhile to manually individually someo of these storms

    display.display(df_post_1900_fixed_interval)

    # %%
    if use_wind:
        df_post_1900_fixed_interval = df_post_1900_fixed_interval[~df_post_1900_fixed_interval['StormID'].isin(problematic_storms_id)]
        # display.display(df_post_1900_fixed_interval)

        # Check if it is fixed
        _ , debug_info = check_time_series(df_post_1900_fixed_interval)
        if len(debug_info) != 0:
            # display.display(debug_info)
            raise ValueError('The time series is not fixed')
        else:
            print('The time series is fixed')
    # %%
    print(df_post_1900_fixed_interval['Time'].dt.hour.unique())
    # display.display(df_post_1900_fixed_interval)

    # %%
    print(f'The number of observations before removing less than 10 observations: {len(df_post_1900_fixed_interval)}')
    df_post_1900_every_3_hour_10more = drop_less_than_n_observation(df_post_1900_fixed_interval)
    print(f'The number of observations after removing less than 10 observations: {len(df_post_1900_every_3_hour_10more)}')
    print(f'Amount lost: {len(df_post_1900_fixed_interval) - len(df_post_1900_every_3_hour_10more)}')

    # %%
    print(df_post_1900_every_3_hour_10more['Time'].dt.hour.unique())
    # display.display(df_post_1900_every_3_hour_10more)

    # %%
    df_with_lag = apply_lag(df_post_1900_every_3_hour_10more, auto_clean=False)
    # display.display(df_with_lag)
    # %%
    df_xyz_lagged = append_unit_sphere_coord(df_with_lag, with_lag=True)
    # display.display(df_xyz_lagged)

    # %%
    append_translational_direction(df_xyz_lagged)
    # display.display(df_xyz_lagged)

    # %%
    with tf.device('/CPU:0'):
        append_translational_speed(df_xyz_lagged, time=time_step)
    # display.display(df_xyz_lagged)

    # %%
    # df_xyz_lagged = append_lambert_conformal_conic(df_xyz_lagged, with_lag=True)
    # display.display(df_xyz_lagged)

    # # %%
    # # Some exploratory data analysis on the added variables
    # plt.hist(df_xyz_lagged['TransSpeed'], bins=50)
    # plt.title('Translational Speed')

    # # %%
    # plt.hist(df_xyz_lagged['TransDir'], bins=50)
    # plt.title('Translational Direction')

    # # %%
    # plt.hist(df_xyz_lagged['LambertX'], bins=50)
    # plt.title('Lambert X')

    # # %%
    # plt.hist(df_xyz_lagged['LambertY'], bins=50)
    # plt.title('Lambert Y')

    # %%
    # Now, I have to store these processed dataframes into a new feather or pickle file
    if use_wind:
        path = '../data/processed/IBTrACS.WP.v04r01.processed.wind.pkl'
    else:
        path = '../data/processed/IBTrACS.WP.v04r01.processed.nowind.pkl'
    df_xyz_lagged.to_pickle(path)

    # %%
    df_test_test = pd.read_pickle('../data/processed/IBTrACS.WP.v04r01.processed.nowind.pkl')

    # %%
    return path

# %%
# This function will normalise the dataset on the given rows
def normalise(df, keys):
    for key in keys:
        min_val = df[key].min()
        max_val = df[key].max()
        df[key+'_norm'] = (df[key] - min_val) / (max_val - min_val)
    return df

def normalise(df):
    return (df - df.min()) / (df.max() - df.min())

def de_normalise(value, max, min):
    return value * (max - min) + min

def unit_sphere_to_lat_lon(x, y, z):
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    if lon < 0:
        lon += 360
    return lat, lon