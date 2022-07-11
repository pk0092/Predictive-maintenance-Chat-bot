# -*- coding: utf-8 -*-

import os
import re
import hgtk
import pickle

import numpy as np
import pandas as pd
import keras

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM
from keras.layers import Embedding

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from layer_utils import Attention


class Continental(object):
    """Most functions are defined under this class."""
    def __init__(self, path):

        self.path = path
        self.equip_name = None
        self.top_k = None
        self.label2idx = None
        self.equip2idx = None
        self.num_classes = None
        self.num_equip = None
        self.num_chars = None
        self.max_seq_len = None

        self.tokenizer = None

    def read_data(self):
        """
        Read data, remove unnecessary columns.
        Returns data as a pd.DataFrame object.
        """
        # 엑셀 파일로부터 데이터 읽기
        data = pd.read_excel(self.path, sheet_name='Sheet1')
        # Data column 확인 (불필요시 삭제해도 무방)
        print(data.columns)

        # 필요한 열 이외 남기고 모두 제거
        columns_to_keep = ['Index', '고장원인전처리', '고장조치 라벨링', 'EquiGroup']
        data = data.filter(items=columns_to_keep)

        # 추후 편의를 위해 column 이름을 교체
        data = data.rename(
            index=str, columns={'고장원인전처리': 'sentences',
                                '고장조치 라벨링': 'labels',
                                'EquiGroup': 'equip'}
        )
        # EquiGroup 빈도 수 확인 (불필요시 삭제해도 무방)
        print('equip counts:\n', data['equip'].value_counts())

        return data

    def filter_by_equip(self, data, equip_name=None):
        """
        Filter rows by EquiGroup keys.
        Unnecessary if all rows must be used.
        """
        self.equip_name = equip_name

        if self.equip_name is None:
            # 모든 관측치 사용
            self.equip_name = 'all'
        else:
            # EquipGroup 값에 따라 일부 관측치만 사용
            equip_mask = data['equip'].isin([self.equip_name])  # 관측치 개수와 동일한 길이의 boolean mask 생성
            data = data[equip_mask].copy()                      # Mask 값이 참(True)인 관측치만 선택

        # 클래스 레이블의 빈도 수 확인 (불필요시 삭제해도 무방)
        print('label counter:\n', data['labels'].value_counts())

        return data

    def filter_by_topk_labels(self, data, k):
        """
        Keep rows with top-k labels, based on frequency.
        """
        self.top_k = k
        label_counts = data['labels'].value_counts()      # 레이블 빈도 수 확인
        labels_to_keep = label_counts.index[:k]           # 가장 많이 등장하는 k개의 레이블 선택
        label_mask = data['labels'].isin(labels_to_keep)  # 관측치 개수와 동일한 길이의 boolean mask 생성
        data = data[label_mask].copy()                    # Mask 값이 참(True)인 관측치만 선택

        return data

    def _sent2char(self, x):
        """Decompose a sentence to characters."""
        x = hgtk.text.decompose(x)
        x = re.sub('\s+', ' ', x)

        return x

    def insert_columns(self, data):
        """
        Insert columns into the dataframe.
        Columns include:
            integer labels (y), integer equip info (x_equip), and character-level input (x_char).
        """

        # Add 'y' column (integers, 0 ~ num_classes - 1)
        self.label2idx = LabelEncoder()
        y = self.label2idx.fit_transform(data['labels'])  # string 타입의 레이블을 정수값으로 변환
        data.insert(data.shape[-1], 'y', y)          # dataframe 맨 뒤에 열 추가
        self.num_classes = len(data['labels'].unique())

        # Add 'x_equip' column (integers, 0 ~ num_equip - 1)
        self.equip2idx = LabelEncoder()
        x_equip = self.equip2idx.fit_transform(data['equip'])  # string 타입의 equip 정보를 정수값으로 변환
        data.insert(data.shape[-1], 'x_equip', x_equip)        # dataframe 맨 뒤에 열 추가
        self.num_equip = len(data['x_equip'].unique())

        # Add 'X_char' column (characters)
        x_char = [self._sent2char(x) for x in data['sentences'].values.tolist()]  # 모든 문장을 character 단위로 변환
        data.insert(data.shape[-1], 'x_char', x_char)                             # dataframe 맨 뒤에 열 추가

        return data

    def save_data(self, data):
        """Save data to pickle file."""
        assert isinstance(data, pd.DataFrame)
        data.to_pickle(
            'datasets/subdata_E-{}_C-{}.pkl'.format(self.equip_name, self.top_k)
        )

    def make_model_inputs(self, subdata):
        """Additional preprocessing."""

        # 자소 단위 데이터를 정수 값으로 변환
        tokenizer = text.Tokenizer(
            num_words=None, filters='', lower=True, split=' ', char_level=True)
        tokenizer.fit_on_texts(subdata['x_char'].values)
        print('number of characters:', len(tokenizer.index_word))
        self.num_chars = len(tokenizer.index_word)

        X_char = tokenizer.texts_to_sequences(subdata['x_char'].values)
        self.max_seq_len = max([len(x) for x in X_char])
        print('max sequence length: ', self.max_seq_len)

        # 최대 문장 길이에 맞추기 (짧은 경우 앞에서부터 0으로 padding, 긴 경우 앞에서부터 제거)
        X_char = sequence.pad_sequences(X_char, maxlen=self.max_seq_len, truncating='pre', padding='pre')

        # Scalar 값으로 표현된 equip 정보를 one-hot vector 형태로 변환
        X_equip = keras.utils.to_categorical(subdata['x_equip'].values)

        # Scalar 값으로 표현된 y를 one-hot vector 형태로 변환
        Y = keras.utils.to_categorical(subdata['y'].values, num_classes=self.num_classes)

        self.tokenizer = tokenizer

        return X_char, X_equip, Y

    def build_model(self, use_equip_info=False):

        # Define character-level input tensor
        char_input = Input(shape=(self.max_seq_len,), dtype='int32')

        # Integer values to 64-dimensional vectors
        h = Embedding(
            input_dim=self.num_chars + 1,
            output_dim=64,
            input_length=self.max_seq_len,
            mask_zero=True)(char_input)

        # LSTM operation
        h = LSTM(
            units=128,
            return_sequences=True,
            unroll=True)(h)

        # Attention layer
        h = Attention(self.max_seq_len)(h)

        if use_equip_info:
            equip_input = Input(shape=(self.num_equip, ), dtype='float32')
            h_equip = Dense(64, activation='relu')(equip_input)
            h = keras.layers.concatenate([h, h_equip])
            y = Dense(self.num_classes, activation='softmax')(h)
            model = Model([char_input, equip_input], y)
        else:
            y = Dense(self.num_classes, activation='softmax')(h)
            model = Model(char_input, y)

        # Define optimizer & compile model
        opt = keras.optimizers.Adam()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=[keras.metrics.categorical_accuracy,
                     self.top3_acc,
                     self.top5_acc]
        )

        print(model.summary())

        return model

    def make_new_test_data(self, sentence, equip_info=None):
        """새로운 관측치를 model input 형태에 맞게 전처리."""
        sentence = self._sent2char(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = sequence.pad_sequences(sentence, maxlen=self.max_seq_len, truncating='pre', padding='pre')

        if equip_info is not None:
            equip_info = self.equip2idx.transform([equip_info])
            equip_info = keras.utils.to_categorical(equip_info, num_classes=self.num_equip)

        return sentence, equip_info

    @staticmethod
    def top3_acc(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    @staticmethod
    def top5_acc(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


if __name__ == '__main__':

    # Make data
    path = './datasets/PM_fail&solution_data.xlsx'
    cont = Continental(path)
    top_k = 100
    data = cont.read_data()
    data = cont.filter_by_equip(data, equip_name=None)
    data = cont.filter_by_topk_labels(data, top_k)
    data = cont.insert_columns(data)
    cont.save_data(data)

    # Read data (example)
    # data = pd.read_pickle('datasets/subdata_E-all_C-100.pkl')

    # Make model inputs
    X_char, X_equip, Y = cont.make_model_inputs(subdata=data)

    # Save 'cont' instance (for later usage)
    with open('./history/cont_instance.pkl', 'wb') as f:
        pickle.dump(cont, f)

    # Instantiate model
    model = cont.build_model(use_equip_info=True)  # Equip 정보를 사용하고 싶은 경우에만 True 설정

    # Randomly split data into train & test sets
    train_indices, test_indices = train_test_split(
        np.arange(len(Y)), test_size=0.2, shuffle=True,
        random_state=123123
    )
    X_char_train, X_equip_train, Y_train = X_char[train_indices], X_equip[train_indices], Y[train_indices]
    X_char_test, X_equip_test, Y_test = X_char[test_indices], X_equip[test_indices], Y[test_indices]

    # Set class weights
    cls_wgt = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(np.argmax(Y_train, axis=-1)),
        y=np.argmax(Y_train, axis=-1)
    )

    # Set checkpoint callback
    model_path = 'history/model_top-{}.hdf5'.format(top_k)
    chkpoint = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)

    # Train model
    hist = model.fit(
        [X_char_train, X_equip_train], Y_train,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        class_weight=cls_wgt,
        verbose=2,
        callbacks=[chkpoint]
    )

    # Save model
    np.save('history/hist_top-{}.npy'.format(top_k), hist.history)

    # Load model
    model.load_weights(model_path)

    # Prediction on test set
    Y_pred = model.predict([X_char_test, X_equip_test], batch_size=256)
    cm = confusion_matrix(np.argmax(Y_test, axis=-1), np.argmax(Y_pred, axis=-1))
    print(classification_report(np.argmax(Y_test, axis=-1), np.argmax(Y_pred, axis=-1)))

    # Evaluation on test set
    test_score = model.evaluate([X_char_test, X_equip_test], Y_test, batch_size=256, verbose=2)
    print('test score (top-1, top-3, top5): ({:.3f}, {:.3f}, {:.3f})'.format(*test_score[1:]))

    # Save final predictions (top-5)
    test_table = []
    for test_index, y_pred in zip(test_indices, Y_pred):

        # Get top-5 predictions (labels & probabilities)
        top_k_indices = np.argsort(y_pred)[::-1][:5]                    # Descending order
        top_k_labels = cont.label2idx.inverse_transform(top_k_indices)  # integer label -> original string
        top_k_proba = y_pred[top_k_indices]                             # probabilities

        # Retrieve original text
        sent = data['sentences'][test_index]

        # Retrieve original index
        original_index = data['Index'][test_index]

        # Retrieve true label
        true_label = data['labels'][test_index]

        # Save values to dictionary
        d = {
            'text': sent,
            'index': original_index,
            'true': true_label
        }
        for i, (_, lb, pb) in enumerate(zip(top_k_indices, top_k_labels, top_k_proba)):
            d.update(
                {'top-{}-pred'.format(i + 1): lb,
                 'top-{}-prob'.format(i + 1): pb}
            )

        test_table.append(d)

    test_table = pd.DataFrame(test_table)
    test_table.to_excel('results/test_results.xlsx', index=False)

    # top 100
    # loss, top1 acc, top3 acc, top5 acc
    # [1.9646599202466826, 0.4786316862890516, 0.7061740984076353, 0.7994808566197856]
    '''
                 precision    recall  f1-score   support
          0       0.73      0.72      0.72        71
          1       0.46      0.53      0.50      2161
          2       0.29      0.18      0.22        68
          3       0.40      0.10      0.16        61
          4       0.00      0.00      0.00        24
          5       0.20      0.07      0.10       247
          6       0.00      0.00      0.00        14
          7       0.19      0.14      0.16        21
          8       0.40      0.12      0.18        51
          9       0.00      0.00      0.00        26
         10       0.66      0.58      0.62       122
         11       0.57      0.08      0.15        48
         12       0.47      0.32      0.38        56
         13       0.73      0.22      0.33        37
         14       0.42      0.57      0.48        40
         15       0.65      0.57      0.61       633
         16       0.27      0.02      0.04       132
         17       0.16      0.10      0.12        29
         18       0.77      0.83      0.80       869
         19       0.29      0.42      0.34       353
         20       0.33      0.07      0.12        56
         21       0.00      0.00      0.00        30
         22       0.00      0.00      0.00        25
         23       0.00      0.00      0.00        60
         24       0.00      0.00      0.00        29
         25       0.17      0.03      0.05        34
         26       0.00      0.00      0.00        38
         27       0.00      0.00      0.00        21
         28       0.08      0.01      0.02       101
         29       0.50      0.03      0.06        61
         30       0.17      0.02      0.03        55
         31       0.67      0.11      0.18        38
         32       0.62      0.14      0.22        37
         33       0.64      0.27      0.38        67
         34       0.41      0.06      0.10       425
         35       0.00      0.00      0.00        36
         36       0.33      0.21      0.26        19
         37       0.34      0.28      0.30       156
         38       0.83      0.64      0.72        94
         39       0.38      0.15      0.22        33
         40       0.29      0.13      0.18       374
         41       0.33      0.37      0.35       796
         42       0.31      0.26      0.28       350
         43       0.42      0.45      0.44        49
         44       0.51      0.73      0.60      2621
         45       0.51      0.66      0.58       441
         46       0.35      0.21      0.26        53
         47       0.27      0.37      0.32        75
         48       0.40      0.08      0.13        26
         49       0.33      0.04      0.07        27
         50       0.29      0.06      0.09       181
         51       0.10      0.05      0.07       105
         52       0.35      0.25      0.29        24
         53       0.90      0.11      0.20        81
         54       0.22      0.11      0.14        19
         55       0.33      0.04      0.07        25
         56       0.38      0.58      0.46        86
         57       0.22      0.14      0.17        43
         58       0.45      0.35      0.39        84
         59       0.47      0.47      0.47        19
         60       0.32      0.16      0.21        63
         61       0.58      0.49      0.53        89
         62       0.00      0.00      0.00        31
         63       0.00      0.00      0.00        31
         64       0.50      0.70      0.59       251
         65       0.11      0.03      0.05        33
         66       0.00      0.00      0.00        26
         67       0.00      0.00      0.00        27
         68       0.54      0.63      0.58      2186
         69       0.26      0.13      0.17       123
         70       0.45      0.44      0.44      2558
         71       0.00      0.00      0.00        89
         72       0.33      0.03      0.06        33
         73       0.51      0.49      0.50       162
         74       0.27      0.13      0.18       252
         75       0.00      0.00      0.00        17
         76       0.00      0.00      0.00        24
         77       0.00      0.00      0.00        29
         78       0.00      0.00      0.00        28
         79       0.27      0.06      0.10        65
         80       0.00      0.00      0.00        26
         81       0.12      0.06      0.08        93
         82       0.00      0.00      0.00        30
         83       0.50      0.07      0.12        90
         84       0.21      0.15      0.18        52
         85       0.00      0.00      0.00        40
         86       0.00      0.00      0.00        32
         87       0.30      0.20      0.24       150
         88       0.41      0.73      0.53      1134
         89       0.33      0.05      0.09        79
         90       0.52      0.28      0.36        50
         91       0.70      0.83      0.76       748
         92       0.50      0.77      0.61        93
         93       0.30      0.24      0.26       268
         94       0.23      0.25      0.24        51
         95       0.36      0.42      0.39       443
         96       0.18      0.13      0.15        15
         97       0.00      0.00      0.00        29
         98       0.25      0.03      0.05        40
         99       0.00      0.00      0.00        37
avg / total       0.45      0.48      0.44     21574
    '''