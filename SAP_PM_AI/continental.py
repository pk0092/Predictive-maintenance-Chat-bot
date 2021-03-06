
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
