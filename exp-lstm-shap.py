import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, TimeDistributed, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime, timedelta
import tensorflow.keras.backend as K
from d_class import exphydro
from hydrodata import DataforIndividual
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
import loss
import shap
import matplotlib.pyplot as plt
import matplotlib
import sys
from sklearn.model_selection import KFold, train_test_split
import loss

matplotlib.use('Agg')  # 使用非交互式后端
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import r2_score, mean_squared_error

# GPU设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# 日志设置
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
tf.compat.v1.experimental.output_all_intermediates(True)

# -------------------- 数据加载与预处理 --------------------
working_path = 'D:\\MyGIS\\hydro_dl_project'
results_dir = Path(working_path) / ("results30")
os.makedirs(results_dir, exist_ok=True)

# 创建K折验证结果目录
kfold_dir = os.path.join(results_dir, "kfold_results")
os.makedirs(kfold_dir, exist_ok=True)

attrs_path = 'D:\\MyGIS\\hydro_dl_project\\camels_data\\datafor_rm\\attributedata671.CSV'
basin_id = ['12375900',

]
basin_id = [bid.zfill(8) for bid in basin_id]
first_basin_id = basin_id[0]  # 保存首个流域ID用于标记

testing_start = '2000-10-01'
testing_end = '2010-09-30'
all_list = []
obs_full_list = []
dates_full = []

for bid in basin_id:
    hydrodata = DataforIndividual(working_path, bid).load_data()
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]

    single_basin_id = bid[1:] if bid.startswith('0') else bid
    static_x = pd.read_csv(attrs_path).set_index('gauge_id')
    row_idx = static_x.index.get_loc(int(single_basin_id))
    local_static = np.array(static_x)[row_idx, :]
    static_rep = np.expand_dims(local_static, 0).repeat(test_set.shape[0], axis=0)

    # 合并数据，但去掉流量列（第6列，索引5）
    # 首先获取除流量外的其他动态特征（前5列）
    dynamic_features = test_set.values[:, :5]  # 只取前5列气象数据

    # 然后合并动态特征和静态属性
    result_test = np.concatenate((dynamic_features, static_rep), axis=-1)

    # 现在result_test包含5列气象数据 + 27列静态属性 = 32列
    all_list.append(result_test)

    obs_full_list.append(test_set.iloc[:, -1].values)  # 流量观测值
    dates_full.extend(test_set.index)

sum_result_test = np.concatenate(all_list, axis=0)
obs_full = np.concatenate(obs_full_list)
dates_full = pd.DatetimeIndex(dates_full)


# -------------------- 原模型预测 --------------------
def create_model(input_shape, seed=None):
    xd_input = Input(shape=input_shape, batch_size=107, name='Input_xd')
    hydro_out = exphydro(mode='normal', name='Regional_dPL_LSTM')(xd_input)
    return Model(inputs=xd_input, outputs=hydro_out)


def generate_test_data(data, wrap_length, step=31):
    num_samples = (data.shape[0] - wrap_length) // step + 1
    x = np.empty((num_samples, wrap_length, data.shape[1]))
    y = np.empty((num_samples, wrap_length, 1))
    for i in range(num_samples):
        start = i * step
        end = start + wrap_length
        x[i] = data[start:end, :]
        y[i, :, 0] = data[start:end, -1]
    return x, y


wrap_length = 365
step = 31
model = create_model((wrap_length, sum_result_test.shape[1]))
save_path = f"{working_path}/results2/{basin_id[0]}_exp.h5"
model.load_weights(save_path)


# 修改重构函数以处理边界情况
def reconstruct_series(windows, step, total_len):
    series_sum = np.zeros(total_len)
    count = np.zeros(total_len)
    num_samps, wl, _ = windows.shape

    for i in range(num_samps):
        st = i * step
        end = min(st + wl, total_len)  # 确保不超过总长度
        actual_len = end - st  # 实际需要填充的长度

        if actual_len > 0:
            # 只取窗口的前actual_len个点
            series_sum[st:end] += windows[i, :actual_len, 0]
            count[st:end] += 1

    count[count == 0] = 1
    return series_sum / count


pred_full_list = []
for bid, data_part in zip(basin_id, all_list):
    test_x, _ = generate_test_data(data_part, wrap_length, step)
    orig_pred = model.predict(test_x, batch_size=None)
    pred_full = reconstruct_series(orig_pred, step, len(data_part))
    pred_full_list.append(pred_full)

pred_full = np.concatenate(pred_full_list)

# -------------------- 误差模型数据划分 --------------------
err_full = (obs_full - pred_full).reshape(-1, 1)

lstm_train_start = '2000-10-01'
lstm_train_end = '2006-09-30'
lstm_test_start = '2006-10-01'
lstm_test_end = '2010-09-30'

mask_train = (dates_full >= lstm_train_start) & (dates_full <= lstm_train_end)
mask_test = (dates_full >= lstm_test_start) & (dates_full <= lstm_test_end)

# 修改：去掉流量特征（最后一列）
train_data = sum_result_test[mask_train, :-1]  # 去掉最后一列（流量）
test_data = sum_result_test[mask_test, :-1]  # 去掉最后一列（流量）
err_train = err_full[mask_train]
err_test = err_full[mask_test]


def generate_error_windows(error_series, wrap_length, step):
    num_samples = (len(error_series) - wrap_length) // step + 1
    y_err = np.zeros((num_samples, wrap_length, 1))
    for i in range(num_samples):
        start = i * step
        end = start + wrap_length
        y_err[i] = error_series[start:end].reshape(-1, 1)
    return y_err


train_y_err = generate_error_windows(err_train, wrap_length, step)
test_y2_err = generate_error_windows(err_test, wrap_length, step)
train_x, train_y = generate_test_data(train_data, wrap_length, step)
test_x2, test_y2 = generate_test_data(test_data, wrap_length, step)


def create_error_model(input_shape):
    m = Sequential(name="Enhanced_Error_LSTM")
    m.add(LSTM(32, return_sequences=True, input_shape=input_shape,
               kernel_regularizer=regularizers.l2(0.005)))
    m.add(BatchNormalization())
    m.add(TimeDistributed(Dense(16, activation='relu')))
    m.add(Dropout(0.1))
    m.add(TimeDistributed(Dense(1)))
    return m


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def r_square(y_true, y_pred):
    # 展平张量（处理序列输出）
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # 计算总平方和
    total_error = tf.reduce_sum(tf.square(y_true_flat - tf.reduce_mean(y_true_flat)))

    # 计算残差平方和
    unexplained_error = tf.reduce_sum(tf.square(y_true_flat - y_pred_flat))

    return 1 - (unexplained_error / (total_error + tf.keras.backend.epsilon()))


# 新增：计算NSE的函数
def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    计算Nash-Sutcliffe效率系数
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算分子（预测值与观测值之差的平方和）
    numerator = np.sum((y_true - y_pred) ** 2)

    # 计算分母（观测值与观测均值之差的平方和）
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # 计算NSE
    nse = 1 - (numerator / denominator)

    return nse


# -------------------- K折交叉验证训练 --------------------
print("\n" + "=" * 50)
print("开始K折交叉验证训练")
print("=" * 50 + "\n")

# K折参数配置
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 保存K折验证结果
fold_metrics = []
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x)):
    print(f"\n{'=' * 50}")
    print(f"训练第 {fold + 1}/{n_splits} 折")
    print(f"{'=' * 50}")

    # 创建当前折的模型
    fold_model = create_error_model((wrap_length, train_x.shape[2]))
    fold_model.compile(
        optimizer='adam',
        loss=root_mean_squared_error,
        metrics=[root_mean_squared_error, MeanSquaredError(name='mse'), r_square]
    )

    # 划分训练和验证数据
    X_train_fold, X_val_fold = train_x[train_idx], train_x[val_idx]
    y_train_fold, y_val_fold = train_y_err[train_idx], train_y_err[val_idx]

    # 设置模型检查点回调
    fold_checkpoint = ModelCheckpoint(
        filepath=os.path.join(kfold_dir, f"fold{fold + 1}_weights.h5"),
        save_weights_only=True,
        monitor='val_root_mean_squared_error',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # 设置回调函数
    fold_callbacks = [
        EarlyStopping(
            monitor='val_root_mean_squared_error',
            patience=10,  # 增加耐心值从10->20
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_root_mean_squared_error',
            factor=0.5,
            patience=5,  # 增加耐心值从5->10
            min_lr=1e-6
        ),
        fold_checkpoint
    ]

    # 训练模型
    fold_history = fold_model.fit(
        X_train_fold, y_train_fold,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=fold_callbacks,
        verbose=1
    )

    # 保存训练历史
    fold_histories.append(fold_history)

    # 在验证集上评估模型
    fold_results = fold_model.evaluate(X_val_fold, y_val_fold, batch_size=32, verbose=1)

    # 保存性能指标
    fold_metrics.append({
        'fold': fold + 1,
        'val_rmse': fold_results[1],
        'val_mse': fold_results[2],
        'val_r2': fold_results[3],
        'epochs': len(fold_history.history['loss'])
    })


    # 绘制本折损失曲线
    def plot_loss_curves(history, save_path):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='训练损失 (RMSE)', color='blue', alpha=0.7)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='验证损失 (RMSE)', color='red', alpha=0.7)
        plt.title('最终模型训练损失曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('均方根误差 (RMSE)')
        plt.legend()
        plt.grid(True)
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_path = f"{save_path.split('.')[0]}_{timestamp}.png"

        plt.savefig(unique_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存至: {unique_path}")


    fold_loss_path = os.path.join(kfold_dir, f"fold{fold + 1}_loss.png")
    plot_loss_curves(fold_history, fold_loss_path)

# 创建并保存K折验证汇总
metrics_df = pd.DataFrame(fold_metrics)
avg_metrics = {
    'fold': '平均',
    'val_rmse': metrics_df['val_rmse'].mean(),
    'val_mse': metrics_df['val_mse'].mean(),
    'val_r2': metrics_df['val_r2'].mean(),
    'epochs': metrics_df['epochs'].mean()
}

summary_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
summary_path = os.path.join(kfold_dir, "kfold_validation_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nK折验证汇总已保存至: {summary_path}")
print(summary_df)

# 确定最佳折
best_fold_idx = metrics_df['val_rmse'].idxmin()
best_fold = metrics_df.loc[best_fold_idx, 'fold']

print(f"\n{'=' * 50}")
print(f"基于K折验证结果，第 {best_fold} 折表现最佳")
print(f"使用第 {best_fold} 折的权重训练最终模型")
print(f"{'=' * 50}")

# -------------------- 使用最佳折训练最终模型 --------------------
error_model = create_error_model((wrap_length, train_x.shape[2]))
error_model.compile(
    optimizer='adam',
    loss=root_mean_squared_error,
    metrics=[root_mean_squared_error, MeanSquaredError(name='mse'), r_square]
)

# 加载最佳折的权重
best_fold_weights = os.path.join(kfold_dir, f"fold{best_fold}_weights.h5")
error_model.load_weights(best_fold_weights)

# 划分训练集和验证集 (90%训练, 10%验证)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    train_x, train_y_err,
    test_size=0.1,
    random_state=42
)

# 使用所有训练数据进一步微调模型，并添加验证集
final_callbacks = [
    EarlyStopping(
        monitor='val_root_mean_squared_error',
        patience=10,  # 增加耐心值从15->25
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_root_mean_squared_error',
        factor=0.5,
        patience=5,  # 增加耐心值从8->12
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath=os.path.join(results_dir, f"final_error_model_{first_basin_id}_weights.h5"),
        save_weights_only=True,
        monitor='val_root_mean_squared_error',
        mode='min',
        save_best_only=True,
        verbose=1
    )
]

history = error_model.fit(
    X_train_final, y_train_final,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=final_callbacks,
    verbose=1
)
final_weights_path = os.path.join(results_dir, f"final_trained_weights_{first_basin_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
error_model.save_weights(final_weights_path)
print(f"最终训练权重已保存至: {final_weights_path}")

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
kfold_run_dir = os.path.join(kfold_dir, f"run_{run_timestamp}")
os.makedirs(kfold_run_dir, exist_ok=True)


# 绘制最终模型的损失曲线（包含验证损失）
def plot_loss_curves(history, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='训练损失 (RMSE)', color='blue', alpha=0.7)
    plt.plot(history.history['val_loss'], label='验证损失 (RMSE)', color='red', alpha=0.7)
    plt.title('最终模型训练/验证损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('均方根误差 (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存至: {save_path}")


final_loss_path = os.path.join(results_dir, f"final_model_loss_{first_basin_id}.png")
plot_loss_curves(history, final_loss_path)


metrics_path = os.path.join(results_dir, f"error_model_metrics_{first_basin_id}.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"指标结果已保存至: {metrics_path}")
# 获取静态特征名称从static_x中
static_feature_names = static_x.columns.tolist()
# 水文特征名称（5个）
hydro_feature_names = ['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)']
# 总特征名称（32个，去除flow后）
feature_names = hydro_feature_names + static_feature_names

try:
    background_data = test_x2
    test_sample = test_x2  # 扩大样本量以获得更稳定的结果

    # 创建聚合模型
    aggregated_output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(error_model.output)
    aggregated_model = Model(inputs=error_model.input, outputs=aggregated_output)

    print("开始计算SHAP值...")
    explainer = shap.GradientExplainer(aggregated_model, background_data)
    shap_values = explainer.shap_values(test_sample)

    # 处理SHAP值
    if isinstance(shap_values, list):
        shap_values_arr = np.array(shap_values[0])  # 对多输出模型取第一个输出
    else:
        shap_values_arr = np.array(shap_values)

    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=(0, 1))
    relative_shap = mean_abs_shap / np.sum(mean_abs_shap)
    # 创建SHAP概述图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_arr.reshape(-1, shap_values_arr.shape[-1]),
        features=test_sample.reshape(-1, test_sample.shape[-1]),
        feature_names=feature_names,
        max_display=32,
        plot_type="bar",
        show=False
    )
    plt.title(f"Feature Importance Overview ({first_basin_id})")
    summary_path = os.path.join(results_dir, f"shap_summary_{first_basin_id}.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP概述图已保存至: {summary_path}")

    # 创建SHAP蜂群图
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_arr.reshape(-1, shap_values_arr.shape[-1]),
        features=test_sample.reshape(-1, test_sample.shape[-1]),
        feature_names=feature_names,
        max_display=32,
        plot_type="dot",
        show=False
    )
    beeswarm_path = os.path.join(results_dir, f"shap_beeswarm_{first_basin_id}.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP蜂群图已保存至: {beeswarm_path}")

except Exception as e:
    print(f"SHAP计算失败: {str(e)}")
    import traceback

    traceback.print_exc()

print(f"\n所有结果已保存至目录: {results_dir}")