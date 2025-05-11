# %%
# This work © 2024 by Isaac Leong is licensed under CC BY-NC-SA 4.0

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
from keras import optimizers
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import IPython.display as display

# %%
import feature_engineering as fe

# %%
use_wind = False

# %%
# datapath = fe.create_features(use_wind=use_wind)

# %%
if use_wind:
    data_path = r'../data/processed/IBTrACS.WP.v04r01.processed.wind.pkl'
else:
    data_path = r'../data/processed/IBTrACS.WP.v04r01.processed.nowind.pkl'

# %%
# df = pd.read_pickle('../data/processed/IBTrACS.WP.v04r01.processed.wind.pkl')
df = pd.read_pickle(data_path)
df.dropna(subset=['Lat','Lon'])
display.display(df)

# %%
# We first need to normalise the dataset first, before we can do anything
feature_set = ['Lat', 'Lon', 'TransDir', 'TransSpeed']
if use_wind:
    feature_set += ['wmo_wind', 'wmo_pressure']

df_norm = fe.normalise(df, feature_set)
display.display(df_norm)

# %%
final_set = [feature + '_norm' for feature in feature_set]
df_train = df_norm[final_set]
display.display(df_train)
# %%
window_size = 5
# data_set_size = 1000000

feature_X = []
feature_Y = []

for name, group in df_train.groupby('StormID'):
    # if(len(feature_X) > data_set_size):
    #     break
    group_values = group.values
    group_values = group_values.astype(object)
    group_values = np.insert(group_values, 0, name, axis=1)
    for j in range(len(group) - window_size):
        feature_X.append(group_values[j:j+window_size])
        feature_Y.append(group_values[j+window_size])

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
display.display(len(feature_X), len(feature_Y))

# %%
display.display(feature_X, feature_Y)

# %%
from sklearn.model_selection import train_test_split
# Create the splitting from the storm_ids
X_train, X_val, y_train, y_val = train_test_split(feature_X, feature_Y, test_size=0.2, random_state=42, shuffle=True)

# %%
# Now, we should drop the StormID from X and Y
X_train = np.array([x[:, 1:3] for x in X_train]).astype(np.float64)
y_train = np.array([y[1:3] for y in y_train]).astype(np.float64)
X_val_with_sid = X_val
y_val_with_sid = y_val
X_val = np.array([x[:, 1:3] for x in X_val]).astype(np.float64)
y_val = np.array([y[1:3] for y in y_val]).astype(np.float64)

# display.display(X_train, y_train, X_eval, y_eval)
display.display(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# %%
display.display(X_train[0], y_train[0])

# %%
display.display(X_train[0][-1])

# %%
print(f"Total length of the dataset: {len(feature_X)}")
print(f"Length of the training dataset: {len(X_train)}")
print(f"Length of the validation dataset: {len(X_val)}")
print(f"Checking for no data loss (training + validation = total): {len(X_train) + len(X_val) == len(feature_X)}")

# %%
def compile_model(input_shape, input_length, epoch_number):
    input = keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(input)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(2, activation='relu')(x)

    #-----------------------------------------------------------

    model = keras.Model(input, outputs=output)
    # cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, input_length * epoch_number, alpha=0.0, name=None
    # )

    opt = keras.optimizers.Adam(learning_rate=5e-3)
    model.compile(optimizer=opt, loss='mse', run_eagerly=True) # type: ignore
    return model

epoch_number = 5
model = compile_model(X_train[0].shape, len(X_train), epoch_number)

# %%
model.summary()

# %%
epoch_number = 5
model.fit(X_train, y_train, batch_size=512, epochs=epoch_number, validation_data=(X_val, y_val), verbose=1)

# %%
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs_range = range(epoch_number)

# %%
figure_x_start = 0

plt.subplot(1, 2, 2)
plt.plot(epochs_range[figure_x_start:], loss[figure_x_start:], label='Training Loss')
plt.plot(epochs_range[figure_x_start:], val_loss[figure_x_start:], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
model_name = 'ibtracs_lstm_model_v2_Lat_Lon_100epoch_batch512_LSTM64_DENSE64_DENSE2_lat_lon.keras'

# %%
model.save(f'../models/{model_name}')

# %%
model = keras.models.load_model(f'../models/{model_name}')

# %%
# Save data for reverse the standardisation
min_before_norm = df[feature_set].min()
max_before_norm = df[feature_set].max()
display.display(min_before_norm, max_before_norm)

# %%
# Pick a random track to test with index
track_index = 100
tarack_with_sid = X_val_with_sid[track_index]
test_track_id = tarack_with_sid[0][0]

display.display(tarack_with_sid, test_track_id)

# %%
# Pick with stormID
test_track_id = b'2000305N06136'

# %%
track = X_val[track_index].tolist()
track = [np.array(track[i]) for i in range(len(track))]
track_denorm = []

for index, row in enumerate(track):
    track_denorm.append(np.array([row[i] * (max_before_norm[feature_set[i]] - min_before_norm[feature_set[i]]) + min_before_norm[feature_set[i]] for i in range(len(row))]))

display.display(track, track_denorm)

# %%
track_norm = None
for index, data in df_train.groupby('StormID'):
    if index == test_track_id:
        print(data, type(data))
        track_norm = data

# %%
original_track = []
for index, data in df.iterrows():
    if index[0] == test_track_id:
        # print(data['Lat'], data['Lon'], data['TransDir'], data['TransSpeed'])
        original_track.append(np.array([data['Lat'], data['Lon'], data['TransDir'], data['TransSpeed']]))

original_track = pd.DataFrame(original_track, columns=['Lat', 'Lon', 'TransDir', 'TransSpeed'])

display.display(original_track)

# %%
denorm_track = []
for index, row in track_norm.iterrows():
    denorm_track.append(np.array([
        row['Lat_norm'] * (max_before_norm['Lat'] - min_before_norm['Lat']) + min_before_norm['Lat'],
        row['Lon_norm'] * (max_before_norm['Lon'] - min_before_norm['Lon']) + min_before_norm['Lon'],
        row['TransDir_norm'] * (max_before_norm['TransDir'] - min_before_norm['TransDir']) + min_before_norm['TransDir'],
        row['TransSpeed_norm'] * (max_before_norm['TransSpeed'] - min_before_norm['TransSpeed']) + min_before_norm['TransSpeed']
    ]))

denorm_track = pd.DataFrame(denorm_track, columns=['Lat', 'Lon', 'TransDir', 'TransSpeed'])

display.display(denorm_track)

# %%
# Check if the denormalisation is correct
residuals = denorm_track - original_track
display.display(residuals)  # Should be zero

# %%
def predict_track(model, norm_track):
    predicted_track = norm_track[:5].to_numpy()[:, :2]

    for i in range(len(norm_track) - 5):
        predicted = model.predict(np.array([predicted_track[-5:]]))
        predicted_track = np.append(predicted_track, predicted, axis=0)

    return predicted_track


outcome_track = predict_track(model, track_norm)
print(outcome_track)

# %%
denorm_track = [
    [outcome_track[i][0] * (max_before_norm['Lat'] - min_before_norm['Lat']) + min_before_norm['Lat'],
    outcome_track[i][1] * (max_before_norm['Lon'] - min_before_norm['Lon']) + min_before_norm['Lon']]
    for i in range(len(outcome_track))
]

# %%
actual_full_track_denorm = original_track[['Lat', 'Lon']].values
print(actual_full_track_denorm)

predicted_track_denorm = denorm_track
print(predicted_track_denorm)

# %%
f, a = plt.subplots(1, 2)

a[0].plot(range(len(actual_full_track_denorm)), np.array(actual_full_track_denorm)[:,0], label='Actual Track')
a[0].plot(range(len(predicted_track_denorm)), np.array(predicted_track_denorm)[:, 0], label='Predicted Track')
a[0].set_title('Actual (blue) vs Predicted (orange) Lat')

a[1].plot(range(len(actual_full_track_denorm)), np.array(actual_full_track_denorm)[:,1], label='Actual Track')
a[1].plot(range(len(predicted_track_denorm)), np.array(predicted_track_denorm)[:, 1], label='Predicted Track')
a[1].set_title('Actual (blue) vs Predicted (orange) Lon')

# %%
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

# %%
map = MapTemplate('Predicted Track on the Earth', min_lon=120, max_lon=180)

map.scatter(np.array(actual_full_track_denorm)[:,1].tolist(), np.array(actual_full_track_denorm)[:,0].tolist(), transform=ccrs.PlateCarree(), color='blue', s=7, marker='o')
map.scatter(np.array(predicted_track_denorm)[:, 1].tolist(), np.array(predicted_track_denorm)[:, 0].tolist(), transform=ccrs.PlateCarree(), color='red', s=9, marker='x')

# %%
# Predict multiple tracks given storm ids
def predict_multiple(storm_ids):
    for test_track_id in storm_ids:
        # track = X_val[track_index].tolist()
        # track = [np.array(track[i]) for i in range(len(track))]
        # track_denorm = []

        # for index, row in enumerate(track):
        #     track_denorm.append(np.array([row[i] * (max_before_norm[feature_set[i]] - min_before_norm[feature_set[i]]) + min_before_norm[feature_set[i]] for i in range(len(row))]))

        # display.display(track, track_denorm)

        track_norm = None
        for index, data in df_train.groupby('StormID'):
            if index == test_track_id:
                print(data, type(data))
                track_norm = data

        original_track = []
        for index, data in df.iterrows():
            if index[0] == test_track_id:
                # print(data['Lat'], data['Lon'], data['TransDir'], data['TransSpeed'])
                original_track.append(np.array([data['Lat'], data['Lon'], data['TransDir'], data['TransSpeed']]))

        original_track = pd.DataFrame(original_track, columns=['Lat', 'Lon', 'TransDir', 'TransSpeed'])

        display.display(original_track)

        denorm_track = []
        for index, row in track_norm.iterrows():
            denorm_track.append(np.array([
                row['Lat_norm'] * (max_before_norm['Lat'] - min_before_norm['Lat']) + min_before_norm['Lat'],
                row['Lon_norm'] * (max_before_norm['Lon'] - min_before_norm['Lon']) + min_before_norm['Lon'],
                row['TransDir_norm'] * (max_before_norm['TransDir'] - min_before_norm['TransDir']) + min_before_norm['TransDir'],
                row['TransSpeed_norm'] * (max_before_norm['TransSpeed'] - min_before_norm['TransSpeed']) + min_before_norm['TransSpeed']
            ]))

        denorm_track = pd.DataFrame(denorm_track, columns=['Lat', 'Lon', 'TransDir', 'TransSpeed'])

        display.display(denorm_track)

        residuals = denorm_track - original_track
        display.display(residuals)  # Should be zero

        def predict_track(model, norm_track):
            predicted_track = norm_track[:5].to_numpy()[:, :2]

            for i in range(len(norm_track) - 5):
                predicted = model.predict(np.array([predicted_track[-5:]]))
                predicted_track = np.append(predicted_track, predicted, axis=0)

            return predicted_track


        outcome_track = predict_track(model, track_norm)
        print(outcome_track)

        denorm_track = [
            [outcome_track[i][0] * (max_before_norm['Lat'] - min_before_norm['Lat']) + min_before_norm['Lat'],
            outcome_track[i][1] * (max_before_norm['Lon'] - min_before_norm['Lon']) + min_before_norm['Lon']]
            for i in range(len(outcome_track))
        ]

        actual_full_track_denorm = original_track[['Lat', 'Lon']].values
        print(actual_full_track_denorm)

        predicted_track_denorm = denorm_track
        print(predicted_track_denorm)

        f, a = plt.subplots(1, 2)

        a[0].plot(range(len(actual_full_track_denorm)), np.array(actual_full_track_denorm)[:,0], label='Actual Track')
        a[0].plot(range(len(predicted_track_denorm)), np.array(predicted_track_denorm)[:, 0], label='Predicted Track')
        a[0].set_title('Actual (blue) vs Predicted (orange) Lat')

        a[1].plot(range(len(actual_full_track_denorm)), np.array(actual_full_track_denorm)[:,1], label='Actual Track')
        a[1].plot(range(len(predicted_track_denorm)), np.array(predicted_track_denorm)[:, 1], label='Predicted Track')
        a[1].set_title('Actual (blue) vs Predicted (orange) Lon')

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

        map = MapTemplate(
                f'Predicted Track on the Earth with id: {test_track_id}', 
                min_lon=min(min(np.array(actual_full_track_denorm)[:,1].tolist()), 
                    min(np.array(predicted_track_denorm)[:, 1].tolist()), 120), 
                max_lon=max(max(np.array(actual_full_track_denorm)[:,1].tolist()), 
                    max(np.array(predicted_track_denorm)[:, 1].tolist()), 180),
                min_lat=min(min(np.array(actual_full_track_denorm)[:,0].tolist()),
                    min(np.array(predicted_track_denorm)[:, 0].tolist())),
                max_lat=max(max(np.array(actual_full_track_denorm)[:,0].tolist()),
                    max(np.array(predicted_track_denorm)[:, 0].tolist()))
            )

        map.scatter(np.array(actual_full_track_denorm)[:,1].tolist(), np.array(actual_full_track_denorm)[:,0].tolist(), transform=ccrs.PlateCarree(), color='blue', s=7, marker='o')
        map.scatter(np.array(predicted_track_denorm)[:, 1].tolist(), np.array(predicted_track_denorm)[:, 0].tolist(), transform=ccrs.PlateCarree(), color='red', s=9, marker='x')

# %%
predict_multiple([b'1987288N13168'])