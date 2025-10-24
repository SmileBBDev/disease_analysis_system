from django.db import models

class LungSample(models.Model):
    """
    CSV 스키마에 맞춘 ORM 모델 (DB 테이블: lungapp_lungsample)
    - 서비스 데이터셋(NA 제거 + 양/불 500씩 균형 샘플)을 저장/사용
    """
    patient_id = models.IntegerField(db_index=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    pack_years = models.FloatField(null=True, blank=True)
    radon_exposure = models.CharField(max_length=50)
    asbestos_exposure = models.CharField(max_length=50)
    secondhand_smoke_exposure = models.CharField(max_length=50)
    copd_diagnosis = models.CharField(max_length=50)
    alcohol_consumption = models.CharField(max_length=50)
    family_history = models.CharField(max_length=50)
    lung_cancer = models.CharField(max_length=20, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "lung_samples"  # 실제 MySQL 테이블명
        indexes = [models.Index(fields=["lung_cancer"])]
        verbose_name = "Lung Sample"
        verbose_name_plural = "Lung Samples"

    def __str__(self):
        return f"#{self.id} pid={self.patient_id} cancer={self.lung_cancer}"
