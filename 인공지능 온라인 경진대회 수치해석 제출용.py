import numpy as np
import pandas as pd
import copy
import pickle
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 경고무시
import warnings
warnings.filterwarnings('ignore')


def label_to_numeric(x):  # Label 형변환용 함수 1
    if x == 'normal':  return 0
    if x == 'out':     return 1
    if x == 'in':      return 2
    if x == 'noise':   return 3
    if x == 'other':   return 4


def numeric_to_label(x):  # Label 형변환용 함수 2
    if x == 0:   return 'normal'
    if x == 1:   return 'out'
    if x == 2:   return 'in'
    if x == 3:   return 'noise'
    if x == 4:   return 'other'


def pred_ensemble_1(pred_1, pred_2, proba_1, proba_2):  # 모델 앙상블 함수 1
    pred = []
    proba = []
    for i in range(len(pred_1)):
        if pred_1[i] == pred_2[i]:
            pred.append(pred_1[i])
            proba.append(max(max(proba_1[i]), max(proba_2[i])))
        elif (pred_1[i] == 'in' or pred_1[i] == 'out') and (pred_2[i] != 'in' and pred_2[i] != 'out'):
            pred.append(pred_1[i])
            proba.append(max(proba_1[i]))
        elif (pred_2[i] == 'in' or pred_2[i] == 'out') and (pred_1[i] != 'in' and pred_1[i] != 'out'):
            pred.append(pred_2[i])
            proba.append(max(proba_2[i]))
        else:
            if max(proba_1[i]) > max(proba_2[i]):
                pred.append(pred_1[i])
                proba.append(max(proba_1[i]))
            else:
                pred.append(pred_2[i])
                proba.append(max(proba_2[i]))

    return pred, proba


def pred_ensemble_2(pred_1, pred_2, proba_1, proba_2):  # 모델 앙상블 함수 2
    pred = []
    for i in range(len(pred_1)):
        if pred_1[i] == pred_2[i]:
            pred.append(pred_1[i])
        elif (pred_1[i] == 'in' or pred_1[i] == 'out') and (pred_2[i] != 'in' and pred_2[i] != 'out'):
            pred.append(pred_1[i])
        elif (pred_2[i] == 'in' or pred_2[i] == 'out') and (pred_1[i] != 'in' and pred_1[i] != 'out'):
            pred.append(pred_2[i])
        else:
            if proba_1[i] > proba_2[i]:
                pred.append(pred_1[i])
            else:
                pred.append(pred_2[i])
    return pred


##############################################################################################################################
##############################################################################################################################

# 1. 데이터 전처리
# data loding - 컬럼별 정규화 데이터셋
data = pd.read_csv('train/train.csv')
scaler = MinMaxScaler()
scaler.fit(data[data.columns[1:]])
data[data.columns[1:]] = scaler.transform(data[data.columns[1:]])
file_name = 'model/minmaxscaler.pkl'
joblib.dump(scaler, file_name)

# 전체 데이터셋을 이용한 train set
X = data[data.columns[1:]]
y = data[data.columns[0]]

sm = SMOTE(random_state=100)
X_resampled_all_1, y_resampled_all_1 = sm.fit_resample(X, y)
sm = SMOTE(random_state=100)
X_resampled_all_2, y_resampled_all_2 = sm.fit_resample(X, y)

# test set은 각 라벨별 최소한도(검증의 역할을 할 수 있는 범위에서)로 추출
data_0 = data.iloc[np.where(data['leaktype'] == 'out')].sample(n=100, random_state=100)
data.drop(data_0.index, inplace=True)
data_1 = data.iloc[np.where(data['leaktype'] == 'in')].sample(n=100, random_state=100)
data.drop(data_1.index, inplace=True)
data_2 = data.iloc[np.where(data['leaktype'] == 'normal')].sample(n=20, random_state=100)
data.drop(data_2.index, inplace=True)
data_3 = data.iloc[np.where(data['leaktype'] == 'other')].sample(n=100, random_state=100)
data.drop(data_3.index, inplace=True)
data_4 = data.iloc[np.where(data['leaktype'] == 'noise')].sample(n=100, random_state=100)
data.drop(data_4.index, inplace=True)

data_test = pd.concat([data_0, data_1], axis=0)
data_test = pd.concat([data_test, data_2], axis=0)
data_test = pd.concat([data_test, data_3], axis=0)
data_test = pd.concat([data_test, data_4], axis=0)

X_train = data[data.columns[1:]]
y_train = data[data.columns[0]]
X_test = data_test[data_test.columns[1:]]
y_test = data_test[data_test.columns[0]]

sm = SMOTE(random_state=100)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

X_train = copy.deepcopy(X_resampled)
y_train = copy.deepcopy(y_resampled)

# data loding - 로우별 정규화 데이터셋
data = pd.read_csv('train/train.csv')
scaler = MinMaxScaler()
scaler.fit(data[data.columns[1:]].T)
data[data.columns[1:]] = scaler.transform(data[data.columns[1:]].T).T

# 전체 데이터셋을 이용한 train set
X_tsc = data[data.columns[1:]]
y_tsc = data[data.columns[0]]

sm = SMOTE(random_state=100)
X_resampled_tsc_all_1, y_resampled_tsc_all_1 = sm.fit_resample(X_tsc, y_tsc)
sm = SMOTE(random_state=10)
X_resampled_tsc_all_2, y_resampled_tsc_all_2 = sm.fit_resample(X_tsc, y_tsc)

# test set은 각 라벨별 최소한도(검증의 역할을 할 수 있는 범위에서)로 추출
data_0 = data.iloc[np.where(data['leaktype'] == 'out')].sample(n=100, random_state=100)
data.drop(data_0.index, inplace=True)
data_1 = data.iloc[np.where(data['leaktype'] == 'in')].sample(n=100, random_state=100)
data.drop(data_1.index, inplace=True)
data_2 = data.iloc[np.where(data['leaktype'] == 'normal')].sample(n=20, random_state=100)
data.drop(data_2.index, inplace=True)
data_3 = data.iloc[np.where(data['leaktype'] == 'other')].sample(n=100, random_state=100)
data.drop(data_3.index, inplace=True)
data_4 = data.iloc[np.where(data['leaktype'] == 'noise')].sample(n=100, random_state=100)
data.drop(data_4.index, inplace=True)

data_test = pd.concat([data_0, data_1], axis=0)
data_test = pd.concat([data_test, data_2], axis=0)
data_test = pd.concat([data_test, data_3], axis=0)
data_test = pd.concat([data_test, data_4], axis=0)

X_train_tsc = data[data.columns[1:]]
y_train_tsc = data[data.columns[0]]
X_test_tsc = data_test[data_test.columns[1:]]
y_test_tsc = data_test[data_test.columns[0]]

sm = SMOTE(random_state=100)
X_resampled_tsc, y_resampled_tsc = sm.fit_resample(X_train_tsc, y_train_tsc)

X_train_tsc = copy.deepcopy(X_resampled_tsc)
y_train_tsc = copy.deepcopy(y_resampled_tsc)

##############################################################################################################################
##############################################################################################################################

# 2. ML모델(KNN, XGB, RF 모델 앙상블)
KNN = KNeighborsClassifier(n_jobs=-1)
RF = RandomForestClassifier(n_jobs=-1, random_state=5023)
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                    gamma=0,
                    gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.300000012,
                    max_delta_step=0,
                    max_depth=6, min_child_weight=1, monotone_constraints='()', n_estimators=100, n_jobs=-1,
                    num_parallel_tree=1,
                    objective='multi:softprob', random_state=5023, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
                    subsample=1, tree_method='auto', validate_parameters=1, verbosity=0)

votingC = VotingClassifier(estimators=[('knn', KNN), ('rf', RF), ('xgboost', XGB)], n_jobs=-1, voting='soft')
print("Voting Classifier 학습중...")
votingC.fit(X_resampled_all_1, y_resampled_all_1)

votingCPickle = open('model/votingC.h5', 'wb')
pickle.dump(votingC, votingCPickle)
print("Voting Classifier 저장\n")

#############################################################################################################################
#############################################################################################################################

# 3. DL모델(1dconv)
# 데이터 전처리(DL)

# 컬럼 정규화, train/test data set
X_train_dl = X_train.values
X_test_dl = X_test.values
X_train_dl = np.reshape(X_train_dl, (X_train_dl.shape[0], X_train_dl.shape[1], 1), order='C')
X_test_dl = np.reshape(X_test_dl, (X_test_dl.shape[0], X_test_dl.shape[1], 1), order='C')

y_train_dl = pd.DataFrame(y_train)['leaktype'].apply(label_to_numeric)
y_test_dl = pd.DataFrame(y_test)['leaktype'].apply(label_to_numeric)
y_train_dl = to_categorical(y_train_dl, num_classes=5)
y_test_dl = to_categorical(y_test_dl, num_classes=5)

# 컬럼 정규화, 전체 data set 1
X_train_dl_all_1 = X_resampled_all_1.values
X_train_dl_all_1 = np.reshape(X_train_dl_all_1, (X_train_dl_all_1.shape[0], X_train_dl_all_1.shape[1], 1), order='C')

y_train_dl_all_1 = pd.DataFrame(y_resampled_all_1)['leaktype'].apply(label_to_numeric)
y_train_dl_all_1 = to_categorical(y_train_dl_all_1, num_classes=5)

# 컬럼 정규화, 전체 data set 2
X_train_dl_all_2 = X_resampled_all_2.values
X_train_dl_all_2 = np.reshape(X_train_dl_all_2, (X_train_dl_all_2.shape[0], X_train_dl_all_2.shape[1], 1), order='C')

y_train_dl_all_2 = pd.DataFrame(y_resampled_all_2)['leaktype'].apply(label_to_numeric)
y_train_dl_all_2 = to_categorical(y_train_dl_all_2, num_classes=5)

# 로우 정규화, train/test data set
X_train_dl_tsc = X_train_tsc.values
X_test_dl_tsc = X_test_tsc.values
X_train_dl_tsc = np.reshape(X_train_dl_tsc, (X_train_dl_tsc.shape[0], X_train_dl_tsc.shape[1], 1), order='C')
X_test_dl_tsc = np.reshape(X_test_dl_tsc, (X_test_dl_tsc.shape[0], X_test_dl_tsc.shape[1], 1), order='C')

y_train_dl_tsc = pd.DataFrame(y_train_tsc)['leaktype'].apply(label_to_numeric)
y_test_dl_tsc = pd.DataFrame(y_test_tsc)['leaktype'].apply(label_to_numeric)
y_train_dl_tsc = to_categorical(y_train_dl_tsc, num_classes=5)
y_test_dl_tsc = to_categorical(y_test_dl_tsc, num_classes=5)

# 로우 정규화, 전체 data set 1
X_train_dl_tsc_all_1 = X_resampled_tsc_all_1.values
X_train_dl_tsc_all_1 = np.reshape(X_train_dl_tsc_all_1,
                                  (X_train_dl_tsc_all_1.shape[0], X_train_dl_tsc_all_1.shape[1], 1), order='C')

y_train_dl_tsc_all_1 = pd.DataFrame(y_resampled_tsc_all_1)['leaktype'].apply(label_to_numeric)
y_train_dl_tsc_all_1 = to_categorical(y_train_dl_tsc_all_1, num_classes=5)

# 로우 정규화, 전체 data set 2
X_train_dl_tsc_all_2 = X_resampled_tsc_all_2.values
X_train_dl_tsc_all_2 = np.reshape(X_train_dl_tsc_all_2,
                                  (X_train_dl_tsc_all_2.shape[0], X_train_dl_tsc_all_2.shape[1], 1), order='C')

y_train_dl_tsc_all_2 = pd.DataFrame(y_resampled_tsc_all_2)['leaktype'].apply(label_to_numeric)
y_train_dl_tsc_all_2 = to_categorical(y_train_dl_tsc_all_2, num_classes=5)

#############################################################################################################################
#############################################################################################################################

batch_size = 512
Epoch = 1000
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, min_lr=0.00001)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode='max', patience=75, verbose=1)

# model_1(컬럼 정규화 데이터)

model_1 = tf.keras.models.Sequential()
model_1.add(
    tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1, input_shape=X_train_dl.shape[-2:]))
model_1.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_1.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_1.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_1.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_1.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1))
model_1.add(tf.keras.layers.GlobalMaxPooling1D())
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Dense(32, activation='relu'))
model_1.add(tf.keras.layers.Dropout(0.35))
model_1.add(tf.keras.layers.Dense(5, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model_1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model_1.fit(X_train_dl, y_train_dl, validation_data=(X_test_dl, y_test_dl), epochs=Epoch, batch_size=batch_size,
            callbacks=[earlystopper, reduce_lr])
loss_ = model_1.evaluate(X_test_dl, y_test_dl)
print("model_1 loss :", loss_)

# 재학습1
model_1.fit(X_train_dl_all_1, y_train_dl_all_1, validation_data=(X_test_dl, y_test_dl), epochs=100,
            batch_size=batch_size)
print("model_1 재학습 1번 완료")

# 재학습2
model_1.fit(X_train_dl_all_2, y_train_dl_all_2, validation_data=(X_test_dl, y_test_dl), epochs=100,
            batch_size=batch_size)
print("model_1 재학습 2번 완료")

model_1.save('model/model_1.h5')  # model_1 200번 재학습
print("model_1 저장\n")

# model_2(로우 정규화 데이터)

model_2 = tf.keras.models.Sequential()
model_2.add(tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1,
                                   input_shape=X_train_dl_tsc.shape[-2:]))
model_2.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_2.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_2.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_2.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_2.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1))
model_2.add(tf.keras.layers.GlobalMaxPooling1D())
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Dense(32, activation='relu'))
model_2.add(tf.keras.layers.Dropout(0.35))
model_2.add(tf.keras.layers.Dense(5, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model_2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model_2.fit(X_train_dl_tsc, y_train_dl_tsc, validation_data=(X_test_dl_tsc, y_test_dl_tsc), epochs=Epoch,
            batch_size=batch_size,
            callbacks=[earlystopper, reduce_lr])
loss_ = model_2.evaluate(X_test_dl_tsc, y_test_dl_tsc)
print("model_2 loss :", loss_)

# 재학습1
model_2.fit(X_train_dl_tsc_all_1, y_train_dl_tsc_all_1, validation_data=(X_test_dl_tsc, y_test_dl_tsc), epochs=100,
            batch_size=batch_size)
print("model_2 재학습 1번 완료")

# 재학습2
model_2.fit(X_train_dl_tsc_all_2, y_train_dl_tsc_all_2, validation_data=(X_test_dl_tsc, y_test_dl_tsc), epochs=100,
            batch_size=batch_size)
print("model_2 재학습 2번 완료")

model_2.save('model/model_2.h5')  # model_2 200번 재학습
print("model_2 저장\n")

# model_3(로우 정규화 데이터, class_weight적용)
d_class_weights = {0: 1, 1: 2, 2: 2, 3: 1.5, 4: 1.5}

model_3 = tf.keras.models.Sequential()
model_3.add(tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1,
                                   input_shape=X_train_dl_tsc.shape[-2:]))
model_3.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_3.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_3.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Conv1D(256, 9, padding='causal', activation='relu', strides=1))
model_3.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Conv1D(128, 9, padding='causal', activation='relu', strides=1))
model_3.add(tf.keras.layers.MaxPooling1D(padding='valid'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Conv1D(64, 9, padding='causal', activation='relu', strides=1))
model_3.add(tf.keras.layers.GlobalMaxPooling1D())
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Dense(32, activation='relu'))
model_3.add(tf.keras.layers.Dropout(0.35))
model_3.add(tf.keras.layers.Dense(5, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model_3.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model_3.fit(X_train_dl_tsc, y_train_dl_tsc, validation_data=(X_test_dl_tsc, y_test_dl_tsc), epochs=Epoch,
            batch_size=batch_size,
            class_weight=d_class_weights, callbacks=[earlystopper, reduce_lr])
loss_ = model_3.evaluate(X_test_dl_tsc, y_test_dl_tsc)
print("model_3 loss :", loss_)

# 재학습1
model_3.fit(X_train_dl_tsc_all_1, y_train_dl_tsc_all_1, validation_data=(X_test_dl_tsc, y_test_dl_tsc), epochs=100,
            batch_size=batch_size)
print("model_3 재학습 1번 완료")

# model_3은 1차 재학습 단계에서 충분히 학습하여(편향적인 test set에 대하여 class_weight가 영향을 준것으로 예상) 2차는 진행하지 않음

model_3.save('model/model_3.h5')  # model_3 100번 재학습
print("model_3 저장\n")

#############################################################################################################################
#############################################################################################################################

# 4. Inference - test set 예측 및 submission 생성
votingCPickle = open('model/votingC.h5', 'rb')
votingC = pickle.load(votingCPickle)
model_1 = tf.keras.models.load_model('model/model_1.h5')
model_2 = tf.keras.models.load_model('model/model_2.h5')
model_3 = tf.keras.models.load_model('model/model_3.h5')

# data loding
test = pd.read_csv('test/test.csv')

# 컬럼 정규화 test set
file_name = 'model/minmaxscaler.pkl'
scaler = joblib.load(file_name)
test[test.columns[1:]] = scaler.transform(test[test.columns[1:]])
X_submission_1 = test[test.columns[1:]]

# 로우 정규화 test set
scaler = MinMaxScaler()
scaler.fit(test[test.columns[1:]].T)
test[test.columns[1:]] = scaler.transform(test[test.columns[1:]].T).T
X_submission_2 = test[test.columns[1:]]

# 데이터 전처리 및 predict
# votingC (컬럼정규화)
y_pred_votingC = votingC.predict(X_submission_1)
y_proba_votingC = votingC.predict_proba(X_submission_1)

# DL 입력용 전처리과정(votingC 모델은 컬럼이름이 필요해서 먼저처리함)
X_submission_1 = X_submission_1.values
X_submission_2 = X_submission_2.values

# model_1 (컬럼정규화)
X_submission_dl_1 = np.reshape(X_submission_1, (X_submission_1.shape[0], X_submission_1.shape[1], 1), order='C')
y_pred_dl_1 = model_1.predict_classes(X_submission_dl_1)
y_pred_dl_1 = pd.DataFrame(y_pred_dl_1)[0].apply(numeric_to_label)
y_proba_dl_1 = model_1.predict_proba(X_submission_dl_1)

# model_2 (로우정규화)
X_submission_dl_2 = np.reshape(X_submission_2, (X_submission_2.shape[0], X_submission_2.shape[1], 1), order='C')
y_pred_dl_2 = model_2.predict_classes(X_submission_dl_2)
y_pred_dl_2 = pd.DataFrame(y_pred_dl_2)[0].apply(numeric_to_label)
y_proba_dl_2 = model_2.predict_proba(X_submission_dl_2)

# model_3 (로우정규화)
X_submission_dl_3 = np.reshape(X_submission_2, (X_submission_2.shape[0], X_submission_2.shape[1], 1), order='C')
y_pred_dl_3 = model_3.predict_classes(X_submission_dl_3)
y_pred_dl_3 = pd.DataFrame(y_pred_dl_3)[0].apply(numeric_to_label)
y_proba_dl_3 = model_3.predict_proba(X_submission_dl_3)

y_pred_ensemble_1, y_proba_ensemble_1 = pred_ensemble_1(y_pred_dl_1, y_pred_dl_2, y_proba_dl_1, y_proba_dl_2)
y_pred_ensemble_2, y_proba_ensemble_2 = pred_ensemble_1(y_pred_dl_3, y_pred_votingC, y_proba_dl_3, y_proba_votingC)
predictions = pred_ensemble_2(y_pred_ensemble_1, y_pred_ensemble_2, y_proba_ensemble_1, y_proba_ensemble_2)

sample_submission = pd.read_csv('sample_submission.csv')
sorter = list(sample_submission['id'])
pred_df = pd.concat([test['id'], pd.DataFrame(predictions)], axis=1)
resdf = pred_df.set_index('id')
resdf.rename(columns={0: 'leaktype'}, inplace=True)
result = resdf.loc[sorter].reset_index()
result.to_csv("submission/submission.csv", index=False)