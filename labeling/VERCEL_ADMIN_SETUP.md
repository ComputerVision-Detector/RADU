# radu-amber.vercel.app 저장소 연결 — 관리자 작업

대상: https://radu-amber.vercel.app/

## 현재 진단과 완료한 작업

2026-10-25 직접 확인 기준 `https://radu-amber.vercel.app/api/labeling?action=health&version=v2`는 HTTP 200으로 `{"status":"ok","storage":"github","samples":163}`을 반환한다. GitHub 읽기는 되지만 **`dataset_version: "v2"`가 없으므로 V2 배포 완료가 아니다.** 기존 API는 version 쿼리를 무시한다. 기존 토큰을 중복 발급할 필요는 없으며, 새 배포 후 토큰 오류가 발생할 때만 아래 교체 절차를 사용한다.

데이터 브랜치 **RADU/labeling-data**의 기존 `labeling/data/live.json`은 V1이다. 현재 V1 상태는 163건 중 146건 검수 완료, 1건 작성 중, 16건 미작성이다. 새 줄 bbox 라벨은 별도 `labeling/data/live-v2.json`에 V2로 저장한다. V1 파일과 평가 파일은 초기화하지 않는다.

코드는 토큰 누락·만료·접근 거부·요청 한도를 구분한다. 아래 토큰 등록·교체는 health가 토큰 오류를 반환할 때 **해당 Vercel 프로젝트 관리자**가 수행한다. 이 개발 세션에는 Vercel 계정 인증이 연결되어 있지 않아 환경변수나 배포 설정은 직접 수정할 수 없다.

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
- Settings → Git → Production Branch: RADU/labeling.
- Root Directory: labeling.
- Framework: Other (vercel.json의 framework=null 유지).
- Node.js: 22.x, Build Command: npm run build, Output Directory: dist.
- 환경변수를 저장한 뒤 Deployments에서 최신 코드의 Production 배포를 **Redeploy**한다.

환경변수 추가만으로 이미 실행 중인 배포가 갱신되지는 않는다. 새 배포가 Ready가 되고 `radu-amber.vercel.app`에 연결되었는지 확인한다.

## 4. 연결 확인

최신 코드 배포 후 다음 URL을 연다.

https://radu-amber.vercel.app/api/labeling?action=health&version=v2

정상이면 HTTP 200으로 다음 형태를 반환한다.

```json
{"status":"ok","dataset_version":"v2","storage":"github","samples":163}
```

`action=list&version=v2` 응답은 V2 bbox 상태를 반환해야 한다. 최초 시작 시 163개 모두 미작성이며, 이미 라벨링이 시작되었다면 저장된 상태를 그대로 유지한다. 화면에는 V1/V2 선택기가 있어야 한다. V1은 기존 146 verified / 1 draft / 16 unlabeled를 유지해야 하며 V2 초기화를 위해 V1 파일을 복사하거나 덮어쓰지 않는다.

이 검사는 저장소 읽기를 확인한다. 쓰기까지 확인하려면 **V2 화면에서** 이미지 하나를 맡은 뒤 작업 반납을 실행한다. 이는 V2 상태 파일이 없으면 최초 생성한다. 실제 정답에 시험 문구나 bbox를 저장하지 않는다. V1 `live.json`의 Git blob SHA가 전후 같아야 한다.

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
