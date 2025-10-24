from django.urls import path
from . import views

app_name = "lungapp"

urlpatterns = [
    path("test/", views.index, name="test"),  # 메인 화면
    # 필요시 다른 엔드포인트 추가
    # path("train/", views.train, name="train"),
    # path("predict/", views.predict, name="predict"),
]
