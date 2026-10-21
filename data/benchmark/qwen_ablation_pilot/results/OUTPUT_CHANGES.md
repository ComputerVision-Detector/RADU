# 출력 변경 사례

**아래는 모델 간 출력 차이이며, 정답 판정이나 정확도 개선 사례가 아닙니다.**

## hf-0081

변경된 줄 17개. 전체 차이는 line_changes.csv에서 확인합니다.

### ocr-text-0001

Paddle:
```text
- 315 평균 제품 50자 (LUSE)
```

Qwen:
```text
- 최소 평균 제곱 오차 (LMSE)
```

### ocr-text-0002

Paddle:
```text
- 모든 점을 해소로 오류를 지낸하고 굳쓰 오류를 밝히는 검신을 찾는 알고리즘
```

Qwen:
```text
- 모든 점을 해소로 오류를 제거하고 최소 오류를 범하는 직선을 찾는 알고리즘
```

### ocr-text-0003

Paddle:
```text
- 허프 변환과 퇴소평균제곱 뜻는 돌다 아웃라이머이
```

Qwen:
```text
- 허프 변환과 최소평균제곱 또는 돌다 아웃라이어에 민감한 긍민해지 많은 개법
```

### ocr-text-0004

Paddle:
```text
민중한 긍민해지 많은 개법
```

Qwen:
```text
- 강인한 극장 - 아웃라이어들 걸레새는 파장을 가진 극장 백 기법
```

### ocr-text-0005

Paddle:
```text
- 강인한 극장
```

Qwen:
```text
- RANSAC
```

### ocr-text-0006

Paddle:
```text
- 아웃라이어들 장례는 대청을 기원 쿠짐 백 기법
```

Qwen:
```text
- 인라이어와 아웃라이어가 섞여 있는 상황에서 인라이어를 찾아 최적 근사하는 기법.
```

### ocr-text-0007

Paddle:
```text
- RA NSA C
```

Qwen:
```text
- 이미 영상에서 선분을 추정하는 문제에서 적용.
```

### ocr-text-0008

Paddle:
```text
- 9다이어와 송국이어가 섞어 있는 송화에서 9각이의들 찾아
```

Qwen:
```text
- 랜덤하게 두 점을 선택하고 두 점을 지나는 직선을 계산
```

## hf-0014

변경된 줄 13개. 전체 차이는 line_changes.csv에서 확인합니다.

### ocr-text-0003

Paddle:
```text
- 궤물 ( label) 또는 참값 (ground truth)
```

Qwen:
```text
- 레이블 (label) 또는 참값 (ground truth)
```

### ocr-text-0004

Paddle:
```text
- 계획참여서 낮은 정보
```

Qwen:
```text
- 가계학습에서 부류 정보
```

### ocr-text-0005

Paddle:
```text
- 황려에 새로는 데이터 집합: 한편 집합
```

Qwen:
```text
- 황련에 사용되는 데이터 집합: 훈련 집합
```

### ocr-text-0006

Paddle:
```text
- 메목에 사용되는 새로운 생물의 검찰 : 테스트 집합.
```

Qwen:
```text
- 예측에 사용되는 새로운 생물의 검찰 : 테스트 집합.
```

### ocr-text-0007

Paddle:
```text
- 윈 솟방습 ( one-shot learning)
```

Qwen:
```text
- 원 속 학습 (one-shot learning)
```

### ocr-text-0008

Paddle:
```text
- 궤물이 있는 생물을 하나만 섬채 참습하는 원숙 참습
```

Qwen:
```text
- 레이블이 있는 생물을 하나만 사용해 학습하는 원속 학습
```

### ocr-text-0009

Paddle:
```text
- 온츠 착습 ( few-shot learning)
```

Qwen:
```text
- 온츠 학습 (few-shot learning)
```

### ocr-text-0010

Paddle:
```text
- 몇 책 선물한 사용되어 참습
```

Qwen:
```text
- 몇 개의 생물만 사용하여 학습
```

## hf-0003

변경된 줄 3개. 전체 차이는 line_changes.csv에서 확인합니다.

### ocr-text-0001

Paddle:
```text
GPT - 005-20b ol "hip" + lello" 是 毕e39
```

Qwen:
```text
GPT - 005-20b 에 "hip", "hello" 를 넣은 경우
```

### ocr-text-0002

Paddle:
```text
Token ID6 [24912, 2504] → shape = [2] → N=2
```

Qwen:
```text
Token IDs [24912, 2504] → shape = [2] → N=2
```

### ocr-text-0007

Paddle:
```text
Embedding. Manlyx [20108, 2000]
```

Qwen:
```text
Embedding Matrix [20108, 2880]
```

## hf-0094

변경된 줄 3개. 전체 차이는 line_changes.csv에서 확인합니다.

### ocr-text-0004

Paddle:
```text
- 윤리의 접구가 몰입 살릿
```

Qwen:
```text
- 윤리의 접구가 몰입 살릿
- 윤리의 문제는 날 (22개로는 새로 왔음)
- 희망을 물려 (탄탄 6개월)
```

### ocr-text-0005

Paddle:
```text
- 윤리의 문제는 날 (22개로는 새로 왔음)
```

Qwen:
```text
- 윤리의 접구가 몰입 살릿
- 윤리의 문제는 날 (22개로는 새로 왔음)
- 희망을 물려 (탄탄 6개월)
```

### ocr-text-0006

Paddle:
```text
- 희망을 물려 (탄탄 6개월)
```

Qwen:
```text
- 윤리의 접구가 몰입 살릿
- 윤리의 문제는 날 (22개로는 새로 왔음)
- 희망을 물려 (탄탄 6개월)
```

## hf-0035

변경된 줄 0개. 전체 차이는 line_changes.csv에서 확인합니다.
