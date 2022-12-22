import numpy as np
import optuna

from sklearn import datasets
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import psutil
import time


def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")

iris = datasets.load_iris()
data = iris.data
target = iris.target

# train 데이터세트와 test 데이터세트로 분리
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1234)

# train 데이터세트와 validation 데이터세트로 분리
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=1234)


print('X_train.shape, X_test.shape, X_validation.shape', X_train.shape, X_test.shape, X_validation.shape)

def objective(trial):

    params = {
        "objective": "multi:softprob",
        "eval_metric":'mlogloss',
        "booster": 'gbtree',
        #'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'gpu_id': 0, # GPU 사용시
        "tree_method": 'exact', 'gpu_id': -1,  # CPU 사용시
        "verbosity": 0,
        'num_class':3,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
        'n_estimators': trial.suggest_int("n_estimators", 1000, 10000, step=100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 1),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),     
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
        # 'num_parallel_tree': trial.suggest_int("num_parallel_tree", 1, 500) 추가하면 느려짐.
    }


    model = xgb.XGBClassifier(**params, random_state = 1234, use_label_encoder = False)

    bst = model.fit(X_train, y_train,eval_set=[(X_validation,y_validation)], early_stopping_rounds=50, verbose=False)
    preds = bst.predict(X_validation)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_validation, pred_labels)
    return accuracy


if __name__ == "__main__":

    train_start = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")


    trial = study.best_trial

    print("  Accuracy: {}".format(trial.value))
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

   
    clf = xgb.XGBClassifier(**study.best_params, random_state = 1234, use_label_encoder = False)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    # pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)

    print("Accuracy: {}".format(accuracy))

    memory_usage("학습하는데 걸린 시간  {:.2f} 분\n".format( (time.time() - train_start)/60))