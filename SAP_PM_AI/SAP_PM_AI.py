import sys


import numpy as np 
import pandas as pd 
import os, re, hgtk, keras, sklearn, pickle, PyQt5

from keras.preprocessing import text, sequence
from keras.models import Model, load_model
from keras.layers import Input, Dense, GRU, LSTM, Embedding

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from layer_utils import Attention
from continental import Continental

from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QTextBrowser, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QCoreApplication, QDateTime
from PyQt5.QtCore import Qt

if __name__ == '__main__':

    with open(u'./Datahub/cont_instance.pkl', 'rb') as f:
        cont = pickle.load(f)
    
    model = cont.build_model(use_equip_info=True)
    
    model_path =u'Datahub/model_top-100.hdf5'
    model.load_weights(model_path)
    k =5
    sentence='a'
    equip_info='b'


class MyApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.datetime = QDateTime.currentDateTime()
        self.retext=''
        self.e='a'
        self.s='b'
        self.initUI()
        
    def initUI(self):

        label1 = QLabel(u'장비 선택', self)
        label2 = QLabel(u'고장 현상', self)
        label3 = QLabel(u'추천 조치 사항', self)
        self.label4 = QLabel(self.datetime.toString(Qt.DefaultLocaleLongDate), self)
        label5 = QLabel('Made by Min Park', self)

        font1 = label1.font()
        font1.setPointSize(20)
        font1.setBold(True)
        font2 = label2.font()
        font2.setPointSize(8)
        font3 = label5.font()
        font3.setPointSize(8)
        font3.setBold(True)
        font4 = label2.font()
        font4.setPointSize(8)
        font4.setBold(True)

        label1.setFont(font1)
        label2.setFont(font1)
        label3.setFont(font1)
        self.label4.setFont(font2)
        label5.setFont(font3)
         
        label1.setGeometry(10,10,300,30)
        label2.setGeometry(10,60,300,30)
        label3.setGeometry(10,130,300,30)
        self.label4.setGeometry(10,325,300,30)
        label5.setGeometry(475,325,300,30)
        
        self.lbl1 = QLabel(self)
        self.lbl1.setText(u"장비를 선택해주세요.")
        self.lbl1.adjustSize()
        self.lbl1.move(250, 10)
        self.cb = QComboBox(self)
        self.cb.addItem(u'선택')
        self.cb.addItem('AOI')
        self.cb.addItem('Cleaneu')
        self.cb.addItem('Conner fill')
        self.cb.addItem('Conveyou')
        self.cb.addItem('Laser marking')
        self.cb.addItem('Loadeu')
        self.cb.addItem('Metal PCB')
        self.cb.addItem('Mounteu')
        self.cb.addItem('Oven')
        self.cb.addItem('Roadrunneu')
        self.cb.addItem('Screen printeu')
        self.cb.addItem('SPI')
        self.cb.addItem(u'자동 삽입기')
        self.cb.move(250, 30)
        self.cb.activated[str].connect(self.equichoose)
        
        self.lbl2 = QLabel(self)
        self.lbl2.setGeometry(250, 60,500,20)
        self.lbl2.setText(u'고장 현상을 작성해주세요.')
        self.lbl2.adjustSize()
        self.qle = QLineEdit(self)
        self.qle.setGeometry(250, 80,255,20)
        self.qle.textChanged[str].connect(self.faileff)
        
        self.input_btn = QPushButton(u'입력',self)
        self.input_btn.pressed.connect(self.input_text)
        self.input_btn.move(520, 70)

        self.tb = QTextBrowser(self)
        self.tb.setAcceptRichText(True)
        self.tb.setOpenExternalLinks(True)
        self.tb.setGeometry(250, 130,330,190)
        
        self.clear_btn = QPushButton(u'초기화',self)
        self.clear_btn.pressed.connect(self.clear_text)
        self.clear_btn.move(520, 95)
        
        pixmap = QPixmap(u'./Continental picture/banner.jpg')
        self.lbl_img = QLabel(self)
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setGeometry(450,5,150,55)

        self.setWindowTitle(u'고장 조치 추천  프로그램')
        self.setWindowIcon(QIcon(u'./Continental picture/emblem1.jpg'))
        self.setGeometry(50,50,600,350)
        self.setFixedSize(600,350)
        self.show()

    def equichoose(self, text):
        if text==u'선택':
            self.lbl1.setText(u'장비를 선택해주세요.')
            self.lbl1.adjustSize()
        else:
            self.e=text
            self.lbl1.setText(u'장비 선택 완료')
            self.lbl1.adjustSize()

    def faileff(self, text):
        self.retext=text.replace(" ","")
        if self.retext=='':
            self.lbl2.setText(u'고장 현상을 작성해주세요.')
            self.lbl2.adjustSize()
        else:
            self.s=text.lower()
            self.lbl2.setText(u'고장 현상 작성 완료')
            self.lbl2.adjustSize()

    def input_text(self):
        if self.e=='a':
            self.lbl1.setText("<font color=red><b>" + u"장비를 선택해주세요." + "</b></font>")
            self.lbl1.adjustSize()

        elif self.s=='b':
            self.lbl2.setText("<font color=red><b>" + u"고장 현상을 입력해주세요." + "</b></font>")
            self.lbl2.adjustSize()

        else:
            self.datetime = QDateTime.currentDateTime()
            dt=self.datetime.toString(Qt.DefaultLocaleLongDate)
            self.label4.setText(dt)
            
            sentence = str(self.s)
            equip_info = str(self.e)
            sentence, equip_info = cont.make_new_test_data(sentence, equip_info)
            y_pred = model.predict([sentence, equip_info])[0]
            
            top_k_indices = np.argsort(y_pred)[::-1][:k]  # Descending order
            top_k_labels = cont.label2idx.inverse_transform(top_k_indices)  # integer label -> original string
            top_k_proba = y_pred[top_k_indices]  # probabilities
            
            aaa=1
            for i, l, p in zip(top_k_indices, top_k_labels, top_k_proba):
                if aaa == 1:
                    self.tb.append(u"추천 조치 사항{1}: {0: <25}".format(l,aaa)+"{0}%".format("%4.2f" % (p*100)))
                    aaa=aaa+1
                    if p < 0.05: 
                        self.tb.append(u"추천할만한 조치 사항이 아닙니다.")


                elif p > 0.01:
                    self.tb.append(u"추천 조치 사항{1}: {0: <25}".format(l,aaa)+"{0}%".format("%4.2f" % (p*100)))
                    aaa=aaa+1



    def clear_text(self):

        self.cb.setCurrentText(u'선택')
        self.lbl1.setText(u'장비를 선택해주세요.')
        self.lbl1.adjustSize()
        self.lbl2.setText(u'고장 현상을 작성해주세요.')
        self.lbl2.adjustSize()
        self.e=='a'
        self.s=='b'
        self.tb.clear()
        self.qle.clear()


        
        

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
