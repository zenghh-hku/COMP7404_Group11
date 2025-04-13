import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc


print('check')

# 加载训练集和测试集
train_data = pd.read_csv("/grp01/saas_lqqu/runxi/lightgbm/train_processed_features.csv")
test_data = pd.read_csv("/grp01/saas_lqqu/runxi/lightgbm/test_processed_features.csv")



train_data = train_data.loc[:, ~train_data.columns.str.contains('\?')]
test_data = test_data.loc[:, ~test_data.columns.str.contains('\?')]

train_label = pd.read_csv("/grp01/saas_lqqu/runxi/lightgbm/train_labels.csv")
test_label = pd.read_csv("/grp01/saas_lqqu/runxi/lightgbm/test_labels.csv")

print('chekc')

train_data = np.array(train_data)
test_data = np.array(test_data)
train_label = np.array(train_label)
test_label = np.array(test_label)


# 转换为DMatrix格式
dtrain = xgb.DMatrix(train_data, label=train_label)
dtest = xgb.DMatrix(test_data, label=test_label)

print('check')

# 基础参数配置
base_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 8,          # 对应num_leaves=255 ≈ 2^8
    "n_estimators": 500,
    "min_child_weight": 100, # 对应min_sum_hessian_in_leaf
    "nthread": 16
}

# 定义两种算法配置
configs = {
    "xgb_exa": {"tree_method": "exact"},  # 预排序算法
    "xgb_his": {"tree_method": "hist"}    # 直方图算法
}

# 训练监控类
class TimeAUCCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        self.start_time = time.time()
        self.time_points = []
        self.auc_values = []
    
    def after_iteration(self, model, epoch, evals_log):
        current_time = time.time() - self.start_time
        self.time_points.append(current_time)
        
        # 从评估日志中提取最新AUC
        if "eval" in evals_log and "auc" in evals_log["eval"]:
            auc = evals_log["eval"]["auc"][-1]
            self.auc_values.append(auc)
        
        return False  # 不停止训练

# 训练和记录
results = {}

for algo_name, algo_params in configs.items():
    # 合并参数
    params = {**base_params, **algo_params}
    
    # 初始化监控器
    monitor = TimeAUCCallback()
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtest, "eval")],
        callbacks=[monitor],
        verbose_eval=False
    )
    
    # 存储结果
    results[algo_name] = {
        "time": monitor.time_points,
        "auc": monitor.auc_values
    }

# 绘制对比曲线
#plt.figure(figsize=(12, 6))

#print(results['exact']['time'])
#print(len(results['exact']['time']))


import pickle

with open('results.pkl', 'wb') as file:
    pickle.dump(results, file)

'''for algo_name, data in results.items():
    plt.plot(
        data["time"], 
        data["auc"], 
        label=f"{algo_name} (Final AUC={data['auc'][-1]:.4f})",
        linewidth=2
    )

plt.xlabel("Training Time (seconds)", fontsize=12)
plt.ylabel("Validation AUC", fontsize=12)
plt.title("XGBoost Algorithms Comparison: Time-AUC Curve", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()'''