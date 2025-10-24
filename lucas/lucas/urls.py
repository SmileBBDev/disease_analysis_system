"""
URL configuration for lucas project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from lucas import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.intro, name = 'intro'), # 기본 루트로 첫 인트로 페이지
    path("home/", views.index, name = 'index'), # /home으로 메인페이지로 이동
    # path('about/', views.about, name='about'),    
    # path('service/', views.service, name='service'),  
    path('contact/', views.contact, name='contact'),  
    path('feature/', views.feature, name='feature'),  # 과거 예측 기록 관리
    path('quote/', views.quote, name='quote'),        # 리포트 다운로드
    path('team', views.team, name="team"),
    path('nopage', views.nopage, name='404'),
    path("lungapp/", include('lungapp.urls')),
    path("predictData/", include('predictData.urls')), # AI 학습
    path("studyData/", include('studyData.urls')), # 진단 예측
    path("viewData/", include('viewData.urls')), # 데이터 관리
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
