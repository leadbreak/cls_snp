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

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
print(device)

random_seed = 41

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

train = pd.read_csv("./data/df_train.csv")
test = pd.read_csv("./data/df_test.csv")
ae = pd.read_csv("./data/ae_values.csv")
train2 = pd.concat([train, ae[:len(train)]], axis=1)
test2 = pd.concat([test, ae[len(train):]], axis=1)

y = torch.LongTensor(train2['class'].values)
X = train2.drop(['id', 'class'], axis=1).to_numpy()
X_test = test2.drop(['id'], axis=1).to_numpy()


xgb_params = {
    'booster': 'gbtree',
    'grow_policy': 'depthwise',
    'max_depth': 4,
    'learning_rate': 0.4,
    'n_estimators': 30,
    'reg_lambda': 100,
    'subsample': 0.9,
    'num_parallel_tree': 1,
    # 'rate_drop': 0.3
}

high = 0
for k in tqdm(range(2, 5+1)) :
    seeds = []
    for _ in range(10000) :
        while True :
            random_seed = np.random.randint(100000)
            if random_seed not in seeds :
                seeds.append(random_seed)
                break
            else :
                continue

        y = train['class'].values
        X = train.drop(['id', 'class'], axis=1).to_numpy()
        X_test = test.drop(['id'], axis=1).to_numpy()

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_seed)

        oof_val_preds = np.zeros((X.shape[0], 3))
        oof_test_preds = np.zeros((X_test.shape[0], 3))

        # OOF
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

            # print('#'*30, f'Fold [{fold+1}/{skf.n_splits}]', '#'*30)

            # train, valid data 설정
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]

            smote = SMOTE(random_state=random_seed)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # 불균형 데이터 가중치 조정 값 => 음성(0) 타깃값 개수 / 양성(1) 타깃값 개수
            _, counts = np.unique(np.array(y_train), return_counts=True)
            scale_weight = counts[0] / counts[1]

            # XGBoost 모델 훈련
            xgb_model = XGBClassifier(
                **xgb_params,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                random_state=random_seed,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train, verbose=False)

            oof_test_preds += xgb_model.predict_proba(X_test) / skf.n_splits
            oof_val_preds[valid_idx] += xgb_model.predict_proba(X_valid)

            # if fold == 1 :
            #     pred = xgb_model.predict(X_test)
            #     break

            #model save
            # xgb_model.save_model(f'./models/new_xgb_{skf.n_splits}_{fold}.json')
            del [[X_train, y_train, X_valid, y_valid, xgb_model]]
            gc.collect()

        #     model score check
        preds = np.argmax(oof_val_preds, axis=1)
        score =  f1_score(y, preds, average="macro")
        if score > high :
            high = score
            text = f'k: {k}, seed : {random_seed}, {score}'
            print(text)

print("="*80)
print()
print(text)
print()
print("="*80)