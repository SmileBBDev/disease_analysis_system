from django.urls import path
from predictData import views

app_name = "predictData" # 예측 모델 생성

urlpatterns = [
    path("predict/", views.data_predict, name="predict"),  # 진단 예측(데이터 훈련) 페이지
    
]
