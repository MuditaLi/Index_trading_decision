import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from warnings import filterwarnings
filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------------------------
feature = pd.read_csv("data/raw/multiasset_feature.csv")
idx = pd.read_csv("data/raw/multiasset_index.csv", index_col='Date')
idx.columns = ['idx_price']

# create lagging features
lag_length = 16
for i in range(1, lag_length):
    idx['lag_' + str(i)] = idx['idx_price'].shift(i)
idx.head(10)

# merge feature and index datasets
dt = feature.merge(idx, how='left', on='Date')

# change some columns formats
cols = ['Fed Balance Sheet', 'US Real Personal Income', 'US Real Personal Income exTrans', 'Adv Retail Sales US exFood Services']
for i in cols:
    dt[i] = dt[i].apply(lambda x: x.replace(",", ""))
    dt[i] = dt[i].apply(lambda x: x.replace(" ", ""))
    dt[i] = dt[i].apply(lambda x: int(float(x)))


# move up price to predict
dt['pred_price'] = dt['idx_price'].shift(-1)
# truncate dataset
dt = dt.iloc[(lag_length-1):, ]
# dt = pd.DataFrame(dt.drop(['Date', 'idx_price'], axis=1))

# split dataset
X = pd.DataFrame(dt.drop(['Date', 'idx_price', 'pred_price'], axis=1))
y = pd.DataFrame(dt['pred_price'])

# normalize features data!!!
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# create train & test sets
X_train = X_scaled[:202, :]
X_test = X_scaled[202:, :]
y_train = y.iloc[:202, ]
y_test = y.iloc[202:, ]

# split out last row as forecast data
# X_forecast = pd.DataFrame(X_test[-1, ]).T
X_forecast = X_test[-1, ]
X_test = X_test[:-1, :]
y_test = y_test[:-1]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_forecast = X_forecast.reshape((1, 1, X_forecast.shape[0]))

# design network
n_steps = X_train.shape[1]
n_features = X_train.shape[2]


def root_mean_squared_error(y_true, y_pred):
    return tensorflow.keras.backend.sqrt(tensorflow.keras.losses.MSE(y_true, y_pred))

# ====================================================================================================================
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss=root_mean_squared_error)
# ====================================================================================================================

# fit the autoencoder model to reconstruct input
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, restore_best_weights=True)
history = regressor.fit(X_train, y_train, epochs=1000, batch_size=8, validation_data=(X_test, y_test), verbose=2,
                        shuffle=False # , callbacks=[early_stop]
                        )

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.title('RMSE Plot')
plt.legend()
plt.show()

# make a prediction
y_hat = regressor.predict(X_test)
y_test_pred = np.concatenate((y_test, y_hat), axis=1)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_hat))
mae = mean_absolute_error(y_test, y_hat)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)

# forecast for next month
y_forecast = regressor.predict(X_forecast)