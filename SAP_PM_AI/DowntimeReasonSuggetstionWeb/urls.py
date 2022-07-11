from django.contrib import admin
from django.urls import include, path
from django.urls import include, path
from django.conf.urls import url
from django.conf import settings


# if settings.DEBUG:
#     import debug_toolbar
#     urlpatterns = [
#         path('__debug__/', include(debug_toolbar.urls)),
#         path('', include(('downtimeML.urls', 'downtimeML'), namespace='downtimeML')),
#         path('admin/', admin.site.urls),
#         url(r'^downtimeML/', include(('downtimeML.urls', 'downtimeML'), namespace='downtimeML')),

#     ] 

urlpatterns = [
    path('', include(('downtimeML.urls', 'downtimeML'), namespace='downtimeML')),
    path('admin/', admin.site.urls),
    url(r'^downtimeML/', include(('downtimeML.urls', 'downtimeML'), namespace='downtimeML')),
     ] 

