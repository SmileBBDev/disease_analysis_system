from django.urls import path
from viewData import views

app_name = "viewData" # 데이터 로드

urlpatterns = [
    path("dataView/", views.data_view, name="dataView"),  # 데이터 조회
    
]
