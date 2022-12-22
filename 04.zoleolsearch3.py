import warnings
warnings.filterwarnings(action='ignore')

import os
import gc
import math
import random
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from tqdm.auto import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from transformers import get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, sampler

from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

device = torch.device('cuda:4') if torch.cuda.is_available() else torch.device('cpu')
print(device)

random_seed = 5833

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(seed=random_seed) # Seed 고정

train = pd.read_csv("./data/df_train5.csv")
test = pd.read_csv("./data/df_test5.csv")

y = torch.LongTensor(train['class'].values)
X = train.drop(['id', 'class', 'class_B', 'class_C'], axis=1).to_numpy()
X_test = test.drop(['id', 'class', 'class_B', 'class_C'], axis=1).to_numpy()


xgb_params = {
    'booster': 'gbtree',
    'grow_policy': 'lossguide',
    'max_depth': 0,
    'learning_rate': 0.3,
    'n_estimators': 50,
    'reg_lambda': 100,
    'subsample': 0.9,
    'num_parallel_tree': 1,
    # 'rate_drop': 0.5
}

high = 0
seeds = []

y = train['class'].values
X = train.drop(['id', 'class', 'class_B', 'class_C'], axis=1).to_numpy()
X_test = test.drop(['id', 'class', 'class_B', 'class_C'], axis=1).to_numpy()

oof_val_preds = np.zeros((X.shape[0], 3))
oof_test_preds = np.zeros((X_test.shape[0], 3))

smote = SMOTE(random_state=random_seed)
X_train, y_train = smote.fit_resample(X, y)
# XGBoost 모델 훈련
xgb_model = XGBClassifier(
    **xgb_params,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=random_seed,
    n_jobs=-1
)

for i in tqdm(range(50000)) :
    random_seed = i    
    
    xgb_model.fit(X_train, y_train, verbose=False)

    oof_test_preds += xgb_model.predict_proba(X_test) 
    oof_val_preds += xgb_model.predict_proba(X)

    gc.collect()

    #     model score check
    preds = np.argmax(oof_val_preds, axis=1)
    score =  f1_score(y, preds, average="macro")
    if score > high :
        high = score
        text = f'seed : {random_seed}, {score}'
        print(text)

print("="*80)
print()
print(text)
print()
print("="*80)