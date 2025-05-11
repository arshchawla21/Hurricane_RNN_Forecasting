# # %% [markdown]
# # # Importing
# # First, we have to import a bunch of libraries

# # %%
# ! pip install pip-tools

# # %%
# ! pip-compile requirements.in

# # %%
# ! pip-sync

# %%
import tensorflow as tf
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import netCDF4
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

import feature_engineering as fe

import IPython.display as display

# %%
# Set the dataset to use
first_dataset = '../data/raw/IBTrACS.WP.v04r01.nc'
second_dataset = '../data/raw/pacific.csv'

dataset = first_dataset
# %%
included_cols = ['sid', 'time', 'lat', 'lon']
if dataset == first_dataset:
    xrds = xr.open_dataset(dataset)
    # dimensions = xrds.dims
    # coords = xrds.coords
    xrds_selected = xrds[included_cols]
    df = xrds_selected.to_dataframe().reset_index()
else:
    df = pd.read_csv(dataset)
    included_cols.append('Date')

df = df[included_cols]
df.rename(columns={'sid': 'StormID', 'time': 'Time', 'lat': 'Lat', 'lon': 'Lon'}, inplace=True)
print(df.head())

# %% [markdown]
# # Data Processing
# 1. Data Cleaning => Drop data from before 1900 and remove storms with less than 15 observations
# 2. Transform Lat and Lon to unit sphere

# Need to do extra processing for the second dataset:
# 1. Convert coordinates with N, S, E, W to coordinates with +/- numeric values
# 2. Combine the Date and Time columns into a single column
if dataset == second_dataset:
    # Function to convert coordinates from N/S/E/W to numeric values
    def convert_coordinate(coord):
        # Ensure the coordinate is a string before processing
        if isinstance(coord, str):
            coord = coord.strip()  # Remove leading/trailing whitespace
            if 'N' in coord or 'E' in coord:
                return float(coord[:-1].strip())  # Convert and keep positive
            elif 'S' in coord or 'W' in coord:
                return -float(coord[:-1].strip())  # Convert to negative value

        return coord

    # Apply the conversion to both the 'Latitude' and 'Longitude' columns
    df['Lat'] = df['Lat'].astype(str).apply(convert_coordinate).astype(float)
    df['Lon'] = df['Lon'].astype(str).apply(convert_coordinate).astype(float)

    # Convert current date and time strings to datetime objects
    def parse_date_time(row):
        # Time 0:00 is represented as 0 in the dataset, need to convert
        time = "000" if str(row["Time"]) == "0" else str(row["Time"])
        hour = time[:1] if len(time) == 3 else time[:2]
        minute = time[-2:]

        date = str(row["Date"])
        year = date[:4]
        month = date[4:6]
        day = date[6:]
        string = year + "/" + month + "/" + day + " " + hour + ":" + minute
        return datetime.strptime(string, "%Y/%m/%d %H:%M")

    df["Time"] = df.apply(parse_date_time, axis=1)
    df = df.drop('Date', axis=1)
    print(df.head())

# %%
def append_unit_sphere_coord(dframe, lag=False):
    dframe['x'] = np.cos(np.radians(dframe.Lat)) * np.cos(np.radians(dframe.Lon))
    dframe['y'] = np.cos(np.radians(dframe.Lat)) * np.sin(np.radians(dframe.Lon))
    dframe['z'] = np.sin(np.radians(dframe.Lat))
    
    if lag:
        dframe['xLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.cos(np.radians(dframe.LonLag1))
        dframe['yLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.sin(np.radians(dframe.LonLag1))
        dframe['zLag1'] = np.sin(np.radians(dframe.LatLag1))

# %%
df_post_1900 = df[df['Time'].dt.year >= 1900]
df_post_1900

# %%
# Group by StormID
df_grouped = df_post_1900.groupby('StormID')

# Drop StormIDs with less than 15 observations
sid_to_drop = []

# Loop through each group
for group in df_grouped:
    if group[1]['Lat'].shape[0] < 15:
        sid_to_drop.append(group[0])

# %%
# Drop the StormIDs with less than 15 data points
df_clean = df_post_1900[~df_post_1900['StormID'].isin(sid_to_drop)]
df_clean

# %%
# Check that all groups have at least 15 records

df_clean_grouped = df_clean.groupby('StormID')

for group in df_clean_grouped:
    if group[1]['Lat'].shape[0] < 15:
        raise ValueError('Group has less than 15 records')
    

# %%
# Create the Lagged Columns
df_clean_grouped_lag = df_clean_grouped.apply(lambda group: group.assign(
    LatLag1=group['Lat'].shift(1),
    LonLag1=group['Lon'].shift(1)
))

# %%
display.display(df_clean_grouped_lag)

# %%
df_clean_grouped_lag = df_clean_grouped_lag.dropna()
display.display(df_clean_grouped_lag)

# %%
fe.append_unit_sphere_coord(df_clean, with_lag=False)
display.display(df_clean)

# %% [markdown]
# # Dataset Preprocessing
# 1. Create Features
# 2. Splitting into train, val, test
# 3. Windowing

# %% [markdown]
# ## Creating Features
# For the project, we shall choose the window size as 5

# %%
print(df_clean_grouped.size())

# %%
window_size = 5
data_set_size = 1000000

feature_X = []
feature_Y = []

for name, group in df_clean_grouped:
    group_values = group[['StormID', 'x', 'y', 'z']].values

    if(len(feature_X) > data_set_size):
        break

    for j in range(len(group) - window_size):
        feature_X.append(group_values[j:j+window_size])
        feature_Y.append(group_values[j+window_size])

# display(feature_X, feature_Y)

# %%
print(len(feature_X), len(feature_Y))

# %%
# Check that the StormID matches in a single feature
def check_data_valid(feature_X, feature_Y, window_size):
    # Check that the StormID matches in the corresponding X and y
    assert len(feature_X) == len(feature_Y) # Check that the length of the two lists are the same
    for i in range(len(feature_X)):
        if feature_X[i][0][0] != feature_Y[i][0]:
            raise ValueError('StormID does not match')
    print("StormID matches in the time series data input")
    
    # Check that the StormID matches in the X
    for x in feature_X:
        for i in range(1, window_size):
            if(x[i][0] != x[i-1][0]):
                raise ValueError('StormID does not match')
    print("StormID matches in the time series data input and its corresponding output")


check_data_valid(feature_X, feature_Y, window_size)

# %%
from sklearn.model_selection import train_test_split
# Create the splitting from the storm_ids
X_train, X_eval, y_train, y_eval = train_test_split(feature_X, feature_Y, test_size=0.2, shuffle=True)

# %%
# Now, we should drop the StormID from X and Y
X_train = np.array([x[:, 1:] for x in X_train]).astype(np.float32)
y_train = np.array([y[1:] for y in y_train]).astype(np.float32)
X_eval_with_sid = X_eval
y_eval_with_sid = y_eval
X_eval = np.array([x[:, 1:] for x in X_eval]).astype(np.float32)
y_eval = np.array([y[1:] for y in y_eval]).astype(np.float32)

# display(X_train, y_train, X_eval, y_eval)
display.display(X_train.shape, y_train.shape, X_eval.shape, y_eval.shape)

# %%
print(f"Total length of the dataset: {len(feature_X)}")
print(f"Length of the training dataset: {len(X_train)}")
print(f"Length of the evaluation dataset: {len(X_eval)}")
print(f"Checking for no data loss (training + eval = total): {len(X_train) + len(X_eval) == len(feature_X)}")

# %%
# import tf_keras
# from tf_keras import layers

# Prev model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.TimeDistributed(layers.Dense(5)),
    layers.LSTM(64, return_sequences=False, activation='relu'),
    layers.Dense(3)
])

# def compile_model(input_shape, input_length, epoch_number):
#     input = keras.Input(shape=input_shape)
#     x = layers.LSTM(64, return_sequences=True)(input)
#     # x = layers.Dense(32, activation='relu')(x)
#     x = layers.LSTM(64, return_sequences=False)(x)
#     x = layers.Dense(64, activation='relu')(x)
#     output = layers.Dense(3, activation='relu')(x)

#     #-----------------------------------------------------------

#     model = keras.Model(input, outputs=output)
#     cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
#         0.01, input_length * epoch_number, alpha=0.1
#     )

#     opt = keras.optimizers.Adam()
#     model.compile(optimizer=opt, loss='mse', run_eagerly=True) # type: ignore
#     return model

# epoch_number = 5
# model = compile_model(X_train[0].shape, len(X_train), epoch_number)

# %%
model.summary()

# %%

# model.compile(optimizer='adam', loss='mse', metrics=['acc', 'mae'])
# model.compile(optimizer='adam', loss='mse')
cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    0.01, input_length * epoch_number, alpha=0.1
)

# %%
# batch_sizes = [2 ** i for i in range(5, 16)]
# learning_rates = [10 ** i for i in range(-5, 0)]
lstm_units = [2 ** i for i in range(2, 10)]
val_losses = []
epochs_run = []
epoch_number = 1000
for lstm_unit in lstm_units:
    model = keras.Sequential([
        layers.LSTM(lstm_unit, return_sequences=False, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.Dense(3)
    ])
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mse')

    print("lstm units: " + str(lstm_unit))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, batch_size=2048, epochs=epoch_number, validation_data=(X_eval, y_eval), verbose=1, callbacks=[callback])
    val_losses.append(min(model.history.history['val_loss']))
    curr_epoch_run = len(model.history.history['val_loss'])
    epochs_run.append(curr_epoch_run)
    model.save('../models/lstm_units/ibtracs_lstm_model_' + str(curr_epoch_run) + 'epoch_batch2048_LSTM' + str(lstm_unit) + '_DENSE3.keras')

# %%
plt.subplot(1, 1, 1)
plt.plot(lstm_units, val_losses)
plt.title('LSTM Units vs Validation Loss')
plt.xlabel("LSTM Units")
plt.ylabel("Validation Loss")
plt.show()

plt.subplot(1, 1, 1)
plt.plot(lstm_units, epochs_run)
plt.title('LSTM Units and Number of Epochs Trained')
plt.xlabel("LSTM Units")
plt.ylabel("Number of Epochs")
plt.show()
# epoch_number = 10
# model.fit(X_train, y_train, batch_size=2048, epochs=epoch_number, validation_data=(X_eval, y_eval), verbose=1)

# %%
pd.set_option('display.float_format', '{:.4g}'.format)

batch_size_df = pd.DataFrame({"LSTM Units": lstm_units, "Validation loss": val_losses, "Epochs": epochs_run})
batch_size_df

# %%
model.history.history

# %%
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs_range = range(epoch_number)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
model.evaluate(X_eval, y_eval)

# %%
model.save('../models/pacific_lstm_model_v2_1000epoch_batch1024_LSTM64_DENSE3.keras')

#%% This is the best performance model so far
model = tf.keras.models.load_model('../models/ibtracs_lstm_model_v5_10epoch_batch1024_LSTM64_DENSE3.keras')
# %% Predict a track
def predict_track(model, track, window_size):
    predictions = []
    
    for i in range(window_size):
        predictions.append(track[i].numpy().tolist())

    print(predictions)
    
    for i in range(len(track) - 2 * window_size):
        input = predictions[i:i+window_size]
        input = tf.convert_to_tensor(input)
        input = tf.expand_dims(input, axis=0)
        predictions.append(model.predict(input)[0].tolist())
    return predictions
# %% Predict a track
# test = track[1:1+window_size][['x', 'y', 'z']].astype(np.float32)
# test = tf.convert_to_tensor(test)
# test = tf.expand_dims(test, axis=0)

# print(test)

# print(test.shape)

# model.predict(test)

possible_ids = []

for i in range(len(X_eval_with_sid)):
    if X_eval_with_sid[i][0][0] not in possible_ids:
        possible_ids.append(X_eval_with_sid[i][0][0])

print(possible_ids)

# %% Perform a single predictions

if dataset == first_dataset:
    prediction_target_id = b'1962206N14140'
else:
    prediction_target_id = 'CP011993'

track = df_clean_grouped.get_group(prediction_target_id)
# predictions = predict_track(model, track, window_size)
# print(predictions)
# track = track[['x', 'y', 'z']].astype(np.float32)
# track = tf.convert_to_tensor(track)
# predict_track = []
# for i in range(0, 90):
#     input = track[i:i+window_size]
#     input = tf.expand_dims(input, axis=0)
#     predict_track.append(model.predict(input)[0])
# input = track[:window_size]
# input = tf.expand_dims(input, axis=0)
# print(model.predict(input)[0])
track = track[['x', 'y', 'z']].astype(np.float32)
track = tf.convert_to_tensor(track)
predicted_track = predict_track(model, track, window_size)

print(predicted_track)

# %% Plot the X
prediction_size = len(predicted_track)
prediction_range = range(window_size, window_size + prediction_size)
# actual_range = range(len(track)) 
# plt.subplot(1, 2, 1)
# plt.plot(track['x'], range(len(track)), label='Actual Track')
# predicted_x = []
# for prediction in predict_track:
#     predicted_x.append(prediction[0])
# plt.plot(predicted_x, prediction_range, label='Predicted Track')
# plt.show()
actual_x = []
actual_y = []
actual_z = []
for i in range(len(track)):
    actual_x.append(track[i][0].numpy())
    actual_y.append(track[i][1].numpy())
    actual_z.append(track[i][2].numpy())

predict_x = []
predict_y = []
predict_z = []
for i in range(len(predicted_track)):
    predict_x.append(predicted_track[i][0])
    predict_y.append(predicted_track[i][1])
    predict_z.append(predicted_track[i][2])

print(actual_x, actual_y, actual_z)
print(predict_x, predict_y, predict_z)

f, a = plt.subplots(3, 1)
a[0].plot(range(len(track)), actual_x, label='Actual Track')
a[0].plot(prediction_range, predict_x, label='Predicted Track')
a[0].set_title('Actual (blue) vs Predicted (orange) X')

a[1].plot(range(len(track)), actual_y, label='Actual Track')
a[1].plot(prediction_range, predict_y, label='Predicted Track')
a[1].set_title('Actual (blue) vs Predicted (orange) Y')

a[2].plot(range(len(track)), actual_z, label='Actual Track')
a[2].plot(prediction_range, predict_z, label='Predicted Track')
a[2].set_title('Actual (blue) vs Predicted (orange) Z')

f.tight_layout()
plt.show()
# %% Plotting on the sphere
# Since the prediction seems legit, we can plot the data on the sphere
# Since the data is in the unit sphere, we can plot the data on the sphere

def unit_sphere_to_lat_lon(x, y, z):
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    if lon < 0:
        lon += 360
    return lat, lon

# Test the Correctness of the function above

p_point5 = unit_sphere_to_lat_lon(predicted_track[0][0], predicted_track[0][1], predicted_track[0][2])
# a_point5 = (df.get_group(b'2024226N27150')['Lat'].iloc[5], df.get_group(b'2024226N27150')['Lon'].iloc[5])

a_point5 = (df_clean_grouped.get_group(prediction_target_id)['Lat'].iloc[5], 
            df_clean_grouped.get_group(prediction_target_id)['Lon'].iloc[5])


print(p_point5, a_point5)

# Some more testing

test_df = pd.DataFrame({
    'Lat' : [42.5],
    'Lon' : [266.9]
})

test_df.Lat = test_df['Lat'].astype(np.float32)
test_df.Lon = test_df['Lon'].astype(np.float32)

append_unit_sphere_coord(test_df, lag=False)
ra, ro = unit_sphere_to_lat_lon(test_df['x'].iloc[0], test_df['y'].iloc[0], test_df['z'].iloc[0])

assert ra - 42.5 < 0.01
assert ro - 266.9 < 0.01

# %% Coverting data for plotting on the sphere

predict_track_lat_lon = pd.DataFrame(columns=['Lat', 'Lon', 'x', 'y', 'z']).astype(np.float32)

for i in range(len(predicted_track)):
    lat, lon = unit_sphere_to_lat_lon(predicted_track[i][0], predicted_track[i][1], predicted_track[i][2])
    predict_track_lat_lon.loc[i] = [lat, lon, predicted_track[i][0], predicted_track[i][1], predicted_track[i][2]]

# Check that they are actually the matching ones
print(predict_track_lat_lon)
display.display(predicted_track)

track_lat_lon = df_clean_grouped.get_group(prediction_target_id)

# %% Plotting the data on the sphere
def MapTemplate(plt_title, min_lat=0, max_lat=60, min_lon=100, max_lon=280):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=180, standard_parallels=(20, 40)))
    # ax.set_extent([100, 280, 0, 60], crs=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    plt.suptitle(plt_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust the rect parameter to leave space for the title
    return ax

map = MapTemplate('Predicted Track on the Earth', min_lon=100, max_lon=300)

map.scatter(track_lat_lon['Lon'].tolist(), track_lat_lon['Lat'].tolist(), transform=ccrs.PlateCarree(), color='blue', s=7, marker='o')
map.scatter(predict_track_lat_lon['Lon'].tolist(), predict_track_lat_lon['Lat'].tolist(), transform=ccrs.PlateCarree(), color='red', s=9, marker='x')

# %%
prediction_target_id
# %%
