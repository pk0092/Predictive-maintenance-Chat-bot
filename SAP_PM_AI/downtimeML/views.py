
import os, pickle
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe

from .models import Category
from modelCreation import Continental

def loadCategories():
    categories = Category.objects.all().order_by('name')

    # template로 보낼 때 공백 문자열 값이 사라지는 문제 해결
    for c in categories:
        c.ename = c.ename.replace(' ','_')

    return categories

def index(request):
    return render(request, 'downtimeML/Index.html', {'categories': loadCategories()})

import tensorflow as tf
logs_path = '/home/dc/pyweb'

# open pk file
with open(os.path.dirname(__file__) + '\cont_instance.pkl' , 'rb') as f:
    print("file is : " + f.name)
    cont = pickle.load(f)

# model 생성하기
model = cont.build_model(use_equip_info=True)

# 미리 학습해둔 모델의 파라미터 불러오기
model.load_weights(os.path.dirname(__file__) + '\model_top-100.hdf5' )

global graph
graph = tf.get_default_graph() 

def getDowntimeReason(request):

    sentence = request.GET.get('contexttext')
    equip_info = request.GET.get('catetoryselect')
    equip_info = equip_info.replace('_',' ') # replace 했던 ' '를 다시 복원

    # 새로운 관측치 전처리 (model input 형태에 맞게)
    sentence, equip_info = cont.make_new_test_data(sentence, equip_info)

    with graph.as_default():
        y_pred = model.predict([sentence, equip_info])[0]
    #y_pred = model._make_predict_function([sentence, equip_info])

    # Get top-5 predictions (labels & probabilities)
    k = 10 # 추천 수
    top_k_indices = np.argsort(y_pred)[::-1][:k]  # Descending order
    top_k_labels = cont.label2idx.inverse_transform(top_k_indices)  # integer label -> original string
    top_k_proba = y_pred[top_k_indices]  # probabilities

    suggestions = []
    for i, l, p in zip(top_k_indices, top_k_labels, top_k_proba):
        s = dict(line = i , name = l , accuracy = round(p*100,2))
        suggestions.append(s)

    return render(request, 'downtimeML/Index.html', {'suggestions':suggestions,'categories': loadCategories()})

def tensorflowtest(request, num1, num2):
    n1 = int(num1)
    n2 = int(num2)

    x =  tf.placeholder(tf.int32)
    y =  tf.placeholder(tf.int32)

    add = tf.add(x, y)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        z = sess.run(add, feed_dict = {x:n1, y:n2})
    return HttpResponse("sum of %s and %s is %s" %(num1, num2, z))
