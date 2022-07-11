from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'^(?P<num1>\d+)/(?P<num2>\d+)/$', views.tensorflowtest, name='tensorflowtest'),

    # path('ajax/load_categories/', views.load_categories, name='ajax_load_categories'),
    path('getDowntimeReason/', views.getDowntimeReason, name='getDowntimeReason'),
]