import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def trainer(df):
    price = df.price

    scaler = MinMaxScaler()

    price = np.asarray(price, dtype='float32')
    min_scale = np.min(price)
    max_scale = np.max(price)
    price -= min_scale
    price /= max_scale
    # price = np.array(df['price'], dtype='float32')
    batch_size=128
    window_size = 100
    shuffle_buffer = 60000

    dataset = windowed_dataset(price, window_size,batch_size, shuffle_buffer)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("lstm01.h5", monitor="val_loss", verbose=1 , save_best_only=True, mode="auto")

    Adam = tf.keras.optimizers.Adam(clipnorm=1.)

    model = Sequential()

    model.add(LSTM(128, input_shape=[None, 1], return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) #과적합 방지를 위한 드랍아웃 비율은 0.5
    model.add(LSTM(64, return_sequences=True)) #LSTM 층  256차원출력+
    model.add(Dropout(0.5)) #드랍아웃 층
    model.add(Dense(1)) #활성화 함수

    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mse'])
    model.summary()

    model.fit(dataset, epochs=10, verbose=1, callbacks=[early_stopping, checkpoint])

    model.save("./drive/My Drive/Colab Notebooks/AGGR/MODEL/model2" +str(item) +".h5")

