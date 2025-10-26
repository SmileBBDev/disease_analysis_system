# view (데이터 조회 views.py)
from lungapp.utils import get_dataset_df
from django.shortcuts import render
from django.contrib import messages
import os
from django.conf import settings
import pandas as pd
import logging
from django.core.paginator import Paginator # 페이징 처리

log = logging.getLogger(__name__)

def data_view(request):
    context = {}
    log.debug("▶ viewData index 진입 | method=%s action=%s", request.method) # [LOG]

    try:
        # 페이지 접속(GET) 시점에 바로 데이터 로드
        df = get_dataset_df()

         # (1) DataFrame을 리스트 형태로 변환
        data_records = df.to_dict(orient="records")

        # (2) 페이지당 표시 개수
        items_per_page = 15

        # (3) Paginator 적용
        paginator = Paginator(data_records, items_per_page)
        page_number = request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)

        # (4) 현재 페이지 데이터만 DataFrame으로 변환 (to_html용)
        # show = df.head(500).copy() if len(df) > 500 else df
        # note = f"총 {len(df)}행 중 500행만 표시" if len(df) > 500 else f"총 {len(df)}행 표시"
        # context["data_html"] = show.to_html(classes="table table-striped table-sm", index=False)
        # context["data_note"] = note
        # messages.success(request, f"데이터 로드 완료: {os.path.basename(settings.LUNG_CANCER_CSV)}")
        show = pd.DataFrame(page_obj.object_list)
        note = f"총 {len(df)}행 중 {items_per_page}개씩 표시 (현재 {page_obj.number}/{paginator.num_pages}페이지)"

        context["data_html"] = show.to_html(classes="table table-striped table-sm", index=False)
        context["data_note"] = note
        context["page_obj"] = page_obj  # 페이지네이션 컨트롤용

        messages.success(request, f"데이터 로드 완료: {os.path.basename(settings.LUNG_CANCER_CSV)}")

    except Exception as e:
        messages.error(request, f"데이터 로드 실패: {e}")
        log.exception("데이터 자동 로드 실패: %s", e)

    return render(request, "viewData/viewdata.html", context)