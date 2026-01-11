import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import optimizers, callbacks
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow.keras.models
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import initializers, constraints, regularizers
from tensorflow.python.keras.layers import Flatten
from tensorflow.keras.layers import Layer, Dense, Lambda, Activation, LSTM, Flatten, Reshape
import tensorflow.keras.backend as K
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)

## Import libraries developed by this study
from d_class import exphydro
from hydrodata import DataforIndividual
import loss
## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = 'D:\\MyGIS\\hydro_dl_project'
attrs_path = 'D:\\MyGIS\\hydro_dl_project\\camels_data\\datafor_rm\\attributedata531.csv'

# List of basins for separate analysis
basin_ids = [
'06853800'
]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
training_start = '1980-10-01'
training_end = '2000-09-30'
nor_start = '1980-10-01'
nor_end = '2000-09-30'
all_list = [ ]
all_list_for_nor = []
for i in range(len(basin_ids)):

    if len(basin_ids[i]) == 7:
        basin_ids[i] = '0' + basin_ids[i]
        print(basin_ids[i])

# Loop through each basin ID and perform the analysis
for basin_id in basin_ids:
    print(f'Processing basin {basin_id}')

    hydrodata = DataforIndividual(working_path, basin_id).load_data()

    # Training and normalizing data
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    nor_set = hydrodata[hydrodata.index.isin(pd.date_range(nor_start, nor_end))]

    a = basin_id
    if a.startswith('0'):
        single_basin_id = a[1:]
    else:
        single_basin_id = a

    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('gauge_id')
    rows_bool = (static_x.index == int(single_basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)

    result = np.concatenate((train_set, local_static_x_for_train), axis=-1)
    nor_set_ = np.concatenate((nor_set, local_static_x_for_train), axis=-1)

    # Prepare data for training
    all_list = [result]
    all_list_for_nor = [nor_set_]

    result_ = all_list[0]
    nor_result_ = all_list_for_nor[0]

    for i in range(len(all_list) - 1):
        result_ = np.concatenate((result_, all_list[i + 1]), axis=0)

    sum_result = result_[:, [0, 1, 2, 3, 4,5]]

    for i in range(len(all_list_for_nor) - 1):
        nor_result_ = np.concatenate((nor_result_, all_list_for_nor[i + 1]), axis=0)

    sum_result1 = nor_result_[:, [0, 1, 2, 3, 4, 5]]

    print("P_mean_std:", np.mean(sum_result[:, 0:1]), np.std(sum_result[:, 0:1]))
    print("Q_mean_std:", np.mean(sum_result[:, -1:]), np.std(sum_result[:, -1:]))

    def generate_train_test(train_set, train_set1, wrap_length):
        train_set_ = pd.DataFrame(train_set)
        train_x_np = train_set_.values[:, :-1]
        train_set1_ = pd.DataFrame(train_set1)
        train_y_np1 = train_set1_.values[:, -1:]
        wrap_number_train = (train_x_np.shape[0] - wrap_length) // 31 + 1

        train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
        train_y1 = np.empty(shape=(wrap_number_train, wrap_length, train_y_np1.shape[1]))

        for i in range(wrap_number_train):
            train_x[i, :, :] = train_x_np[i * 31:(wrap_length + i * 31), :]
            train_y1[i, :, :] = train_y_np1[i * 31:(wrap_length + i * 31), :]

        return train_x, train_y1

    wrap_length = 365  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
    train_x, train_y = generate_train_test(sum_result, sum_result1, wrap_length=wrap_length)

    print(f'The shape of train_x, train_x1, train_y after wrapping by {wrap_length} days are:')
    print(f'{train_x.shape}, {train_y.shape}')

    def create_model(input_xd_shape, seed):
        xd_input_forprnn = Input(shape=input_xd_shape, batch_size=224, name='Input_xd')  # [9,3288,5]
        hydro_output = exphydro(mode='normal', name='Regional_dPL_LSTM')(xd_input_forprnn)
        output_layer = Reshape((-1,))(hydro_output)
        flattened_output = Flatten()(hydro_output)
        model = Model(inputs=xd_input_forprnn, outputs=hydro_output)
        return model


    def train_model(model, train_xd, train_y, ep_number, lrate, save_path):
        # 定义RMSE指标函数
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_true - y_pred)))

        save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                         save_weights_only=True)

        es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                     restore_best_weights=True)

        reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.5, patience=5, verbose=1, mode='max',
                                             min_delta=0.005, cooldown=0, min_lr=lrate / 100)

        tnan = callbacks.TerminateOnNaN()

        # 在metrics中添加RMSE指标
        model.compile(
            loss=loss.nse_loss,
            metrics=[loss.nse_metrics, rmse],  # 添加RMSE指标
            optimizer=tf.keras.optimizers.Adam(learning_rate=lrate)
        )
        history = model.fit(x=train_xd, y=train_y, epochs=ep_number, batch_size=224,
                            callbacks=[save, es, reduce, tnan])
        return history
    # Define save path for each basin
    save_path_ealstm = f'{working_path}/results2/{basin_id}_exp.h5'

    model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]), seed=200)
    model.summary()

    # Train the model for the current basin
    prnn_ealstm_history = train_model(model=model, train_xd=train_x, train_y=train_y, ep_number=150, lrate=0.01,
                                      save_path=save_path_ealstm)

    # Load model weights after training
    model.load_weights(save_path_ealstm)
    print(f"Model for basin {basin_id} trained and saved.")
