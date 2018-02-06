"""vericode_recog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from vericode_recog_tesseract import views

urlpatterns = [
    url(r'^recog', views.vericode_recog_view),
    url(r'^binary', views.binary_image),
    url(r'^clear', views.clear_contour),
    url(r'^train', views.train_data),
    url(r'^crawl', views.crawl_pic),
    url(r'^verify', views.verify_model),
]
