from django.urls import path
from studyData import views

app_name = "studyData" # 데이터 훈련 생성

urlpatterns = [
    path("train/", views.data_study, name="train"),  # 예측 모델 생성 페이지

]
