# 이미지별 전사 정답

같은 번호의 images/hf-XXXX.png를 보고 hf-XXXX.json의 text를 작성합니다. 사람이 검수한 뒤 status를 verified로 바꾸세요. 작성 중은 draft, 미작성은 unlabeled입니다. 미작성 text는 null, 특이사항 없는 notes는 빈 문자열입니다. annotator에 작성자를 적습니다. id는 변경하지 않습니다.

기존 검수 5장(hf-0003, hf-0014, hf-0035, hf-0081, hf-0094)은 원래 정답을 복사했습니다. 나머지 117장은 빈 양식입니다. 파일을 자동 재생성해도 기존 파일은 덮어쓰지 않습니다.

JSON 문자열 안의 따옴표는 \"로, 줄바꿈은 \n으로 적으세요. 모델 결과를 정답으로 복사하지 말고 원본과 독립 대조하세요. 구조 정답을 임의로 text에 추가하지 마세요. 텍스트 기준과 평가용 분할은 작성 전에 통일하는 것이 좋습니다.

병합·규칙 실행 방법은 프로젝트 docs/STRUCTURE_RULES_IMPLEMENTED.md를 참고하세요. 이 폴더의 라벨을 수정하는 것만으로 기존 통합 references.json이나 결과 보고서는 자동 갱신되지 않습니다.

공동 작업·이미지 다운로드·개인 브랜치·PR 방법은 [공동 라벨링 안내](../../../../docs/ANNOTATION_COLLABORATION.md)를 참고하세요.
