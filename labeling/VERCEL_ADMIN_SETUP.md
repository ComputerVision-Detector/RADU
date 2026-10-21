# radu-amber.vercel.app 저장소 연결 — 관리자 작업

대상: https://radu-amber.vercel.app/

## 현재 진단과 완료한 작업

화면은 HTTP 200이지만 `/api/labeling?action=list`가 HTTP 503과 토큰 미설정 오류를 반환한다. 이는 Vercel 서버 함수에 GitHub 토큰이 없다는 의미다. Docker 독립 서버와 달리 Vercel 방식에는 영구 저장소 연결용 토큰이 필요하다.

GitHub 저장소 읽기·쓰기 권한을 확인했고 **RADU/labeling-data** 브랜치의 `labeling/data/live.json`을 초기화했다. 초기 정답 5장 / 미작성 117장, 총 122개를 실제 API로 다시 읽어 확인했다. 기존 평가 파일은 바꾸지 않았다.

코드는 토큰 누락·만료·접근 거부·요청 한도를 구분하도록 보완했다. 아래 토큰 등록과 재배포는 **해당 Vercel 프로젝트 관리자**가 수행해야 한다. 이 개발 세션에는 해당 Vercel 계정 인증이 연결되지 않았다. 운영 전용 새 토큰 발급·등록을 이미 완료한 상태는 아니다.

## 1. GitHub 운영 전용 토큰 생성

[토큰 생성 화면 열기](https://github.com/settings/personal-access-tokens/new?name=HybridOCR-labeling&target_name=ComputerVision-Detector&contents=write)

다음 설정을 화면에서 확인한다.

| 항목 | 설정 |
|---|---|
| Resource owner | ComputerVision-Detector |
| Repository access | Only select repositories → RADU |
| Repository permissions → Contents | Read and write |
| Metadata | Read (자동 포함) |
| Expiration | 팀 운영 기간에 맞춰 설정 |

링크가 값을 미리 채우더라도 Resource owner와 저장소 선택은 직접 확인한다. 계정에 조직 선택 권한이 없으면 조직 관리자가 발급해야 한다. 조직 승인 대기 상태이면 승인 후 사용한다. 이 앱에 Administration·Workflows·조직 전체 관리 권한은 필요 없다.

생성한 토큰 값은 다음 단계 Vercel 서버 환경변수에 입력한다. 코드·채팅·라벨 파일에 넣지 않는다.

## 2. Vercel 환경변수 등록

Vercel에서 **radu-amber.vercel.app이 연결된 프로젝트 → Settings → Environment Variables**로 이동한다.

| Key | Value | 적용 환경 |
|---|---|---|
| LABEL_GITHUB_TOKEN | 방금 생성한 실제 토큰 값 | Production, 필요하면 Preview도 |
| LABEL_GITHUB_REPOSITORY | ComputerVision-Detector/RADU | Production / Preview |
| LABEL_SOURCE_BRANCH | RADU/labeling | Production / Preview |
| LABEL_DATA_BRANCH | RADU/labeling-data | Production / Preview |

토큰에 따옴표나 설명 문장을 붙이지 않는다. 브라우저 공개용 환경변수 이름으로 바꾸지 않는다. Preview만 선택하면 현재 Production 주소에는 반영되지 않는다.

## 3. 최신 코드로 재배포

- 소스 브랜치: RADU/labeling.
- Root Directory: labeling.
- Framework: Other (vercel.json의 framework=null 유지).
- 환경변수를 저장한 뒤 Deployments에서 최신 코드의 Production 배포를 **Redeploy**한다.

환경변수 추가만으로 이미 실행 중인 배포가 갱신되지는 않는다. 새 배포가 Ready가 되고 `radu-amber.vercel.app`에 연결되었는지 확인한다.

## 4. 연결 확인

최신 코드 배포 후 다음 URL을 연다.

https://radu-amber.vercel.app/api/labeling?action=health

정상이면 HTTP 200으로 다음 형태를 반환한다.

```json
{"status":"ok","storage":"github","samples":122}
```

이 검사는 저장소 읽기를 확인한다. 쓰기까지 확인하려면 웹에서 미작성 이미지 하나를 맡은 뒤 작업 반납을 실행한다. 전사 내용을 시험 입력해 실제 정답으로 저장할 필요는 없다. 권한 오류가 없다면 공동 작업 저장도 연결된 것이다.

## 오류 구분

| code | 조치 |
|---|---|
| GITHUB_TOKEN_MISSING | Production에 토큰 등록 → 재배포 |
| GITHUB_TOKEN_INVALID | 만료·오입력 토큰 교체 → 재배포 |
| GITHUB_PERMISSION_DENIED | RADU의 Contents 읽기·쓰기 및 조직 승인 확인 |
| GITHUB_REPOSITORY_UNAVAILABLE | 저장소·기준 브랜치 이름과 토큰 저장소 범위 확인 |
| GITHUB_RATE_LIMITED | GitHub 요청 한도 회복 후 재시도 |
| GITHUB_WRITE_REJECTED | 브랜치 이름·보호 규칙·저장소 쓰기 권한 확인 |

토큰 권한 추가나 교체 후에도 반드시 새 배포를 만든다. 데이터 브랜치 live.json은 기존 공동 작업을 담으므로 초기화하거나 삭제하지 않는다.

## 공식 문서

- [GitHub fine-grained 토큰 생성·권한](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Vercel 환경변수 등록과 재배포](https://vercel.com/docs/environment-variables/managing-environment-variables)
