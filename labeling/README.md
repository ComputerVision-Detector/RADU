# HybridOCR 공동 라벨링 웹

작성자 이름만 입력하고 이미지 원문을 보면서 남은 117장의 정답 전사를 함께 작성하는 웹 앱이다. JSON을 직접 편집하지 않아도 된다. 기존 검수 5장은 참고용으로 보호한다.

- 이미지 목록, 상태·내 작업 필터, 번호·작성자 검색, 전체 진행률.
- 확대·축소·회전·드래그, 별도 큰 이미지 보기.
- 작업 맡기·반납, 25분 임대와 3분 자동 연장.
- 전사·메모, 셀 구분 탭 입력, 임시 저장, 검수 완료, Ctrl/Cmd+S.
- 브라우저에 미저장 입력 백업 및 복구, 버전 충돌 시 덮어쓰기 차단.
- GitHub 서버 저장, 최근 수정 기록, 개별 JSON ZIP·통합 JSON 다운로드.
- 모바일 대응. OCR/Qwen 모델 및 GPU 불필요.

## 다른 사람의 서버·도메인에 바로 전달하기

**[START_HERE.md](START_HERE.md)를 서버 관리자에게 전달하세요.** 이 폴더만으로 Docker 앱과 HTTPS 프록시를 실행할 수 있습니다. 이 독립 서버 방식은 GitHub 토큰·Vercel 계정·별도 DB가 필요 없으며 라벨은 서버 데이터 볼륨에 저장됩니다.

```bash
cp .env.selfhost.example .env
# .env의 LABEL_DOMAIN을 실제 도메인으로 변경
docker compose -f compose.yaml -f compose.domain.yaml up -d --build
```

기존 Nginx 서버에 연결하는 방법, 백업·복원과 서버 실행 조건도 START_HERE.md에 있습니다. 아래 Vercel 안내는 대안이며 독립 서버 배포에 적용하는 설정이 아닙니다.

## Vercel 배포

GitHub 코드 브랜치: **RADU/labeling**. 기존 실험 브랜치 `feat/hybridocr`에서 분기했다.

1. Vercel에서 New Project → GitHub `ComputerVision-Detector/RADU`를 가져온다.
2. Framework Preset은 **Other**, Root Directory는 **labeling**으로 설정한다.
3. 배포할 소스 브랜치를 **RADU/labeling**으로 선택한다. 기본 main을 가져온 경우 Settings → Git에서 Production Branch를 변경하고 해당 브랜치를 배포한다.
4. Node.js **22.x**, Build Command **npm run build**, Output Directory **dist**를 사용한다. `vercel.json`과 `package.json`에 설정되어 있다. 외부 npm 런타임 의존성은 없다.
5. 아래 서버 환경변수를 설정한다. Preview를 사용할 경우 해당 환경에도 설정하고 저장 후 재배포한다.
6. Deploy 후 `.vercel.app` 주소로 접속해 작성자 이름을 입력한다. 커스텀 도메인은 Vercel Settings → Domains에서 연결한다.

| 환경변수 | 값 |
|---|---|
| `LABEL_GITHUB_TOKEN` | 이 저장소에 접근할 수 있는 서버용 GitHub 토큰. 필수. 브라우저 또는 Git에 넣지 않는다. |
| `LABEL_GITHUB_REPOSITORY` | `ComputerVision-Detector/RADU` (기본값) |
| `LABEL_SOURCE_BRANCH` | `RADU/labeling` (기본값) |
| `LABEL_DATA_BRANCH` | `RADU/labeling-data` (기본값) |

토큰은 가능한 한 해당 저장소만 선택한 fine-grained token을 사용하고 **Contents: Read and write** 권한을 부여한다. 조직 정책에서 승인이 필요하면 승인된 토큰을 사용한다. 팀원이 각자 토큰을 입력하는 방식이 아니며 배포 관리자가 서버 환경변수에 한 번 설정한다. 토큰이 없으면 공유 저장 미연결 상태를 표시하고 저장된 것처럼 처리하지 않는다.

Vercel 계정·프로젝트와 환경변수가 연결되기 전에는 실제 공용 URL이 생성된 상태가 아니다. 이 저장소에는 배포 가능한 코드와 설정을 제공한다. GitHub Pages 단독 배포는 서버 API가 없어 공동 저장 기능을 수행할 수 없다. GitHub 저장소를 Vercel에 연결해 배포하는 구성을 사용한다.

공식 설정 참고: [Vercel Git 배포](https://vercel.com/docs/git), [프로젝트 루트 설정](https://vercel.com/docs/monorepos), [Node.js 함수](https://vercel.com/docs/functions/runtimes/node-js).

## 저장 위치와 충돌 처리

Vercel 방식의 원본 상태는 데이터 브랜치 **RADU/labeling-data**의 `labeling/data/live.json`이다. 첫 작업 맡기·저장 시 GitHub API로 브랜치와 파일을 생성한다. 코드 브랜치·main·기존 5장 평가 파일은 수정하지 않는다. 배포가 재시작되어도 GitHub의 상태를 다시 읽는다. Vercel 임시 파일 시스템이나 브라우저 localStorage를 공동 저장소로 사용하지 않는다.

`live.json`은 모든 이미지의 전사·revision·현재 작업자·최근 이력을 포함한다. GitHub 파일 SHA를 조건으로 갱신하며, 서로 다른 이미지에 대한 동시 수정은 최신 상태에 재적용한다. 동일 이미지의 오래된 revision 저장은 409 오류로 막는다. 최근 이력은 이미지당 20개이고 GitHub 커밋에도 상태가 남는다.

기존 `data/benchmark/qwen_ablation_pilot/labels/*.json`은 시작 시점의 라벨이다. 웹 입력은 그 파일들을 매번 직접 갱신하지 않는다. 웹의 **정답 파일 다운로드**에서 최신 개별 JSON ZIP을 내려받아 labels/에 반영하고 PR로 검토한다. 다운로드 결과에는 필요한 id/text/status/annotator/notes와 원래 submitted_id만 포함되며 작업 임대·내부 revision은 제외한다.

`ignore-build.mjs`는 데이터 브랜치의 커밋이 Vercel 앱을 계속 재배포하지 않게 한다. 데이터 브랜치 이름을 바꿀 때 환경변수도 같이 설정한다. GitHub 요청 한도를 초과하거나 연결에 실패하면 저장 실패를 표시하고 입력을 유지한다. 대규모 이용 서비스가 아닌 현재 122장 팀 라벨링용 구성이다.

작성자 이름은 본인 인증이 아니다. 요청한 이름 입력 방식이므로 주소에 접근한 사람은 추가 라벨을 작성할 수 있다. 다른 브라우저가 같은 이름을 입력해도 작업 임대는 브라우저 식별자로 구분한다. 기존 검수 5장 변경은 서버에서 차단한다.

## 이미지

`assets/`에 122장 모두 들어 있다. 최대 변 3200px, WebP quality 85인 웹 읽기용 복사본이며 약 151MiB다. **고정 OCR 비교 입력 PNG를 대체하지 않는다.** 작은 글자를 확대하고 크게 보기로 확인할 수 있다. 압축본으로 판독이 어려운 페이지는 검수 완료하지 말고 메모·임시 저장으로 남긴 뒤 HF 원본과 대조한다.

`data/assets.manifest.json`에 원본 PNG SHA와 웹 이미지 크기·SHA를 보관한다. 원본 PNG나 모델 가중치는 웹 배포에 포함하지 않는다. 이미지 재생성은 저장소 루트에서 실행한다.

```bash
.venv/bin/python labeling/generate_assets.py
```

## 로컬 실행

Node.js 22 이상에서:

```bash
cd labeling
npm run dev
```

http://127.0.0.1:8014/ 에서 확인한다. 로컬/독립 서버 모드는 `.runtime/state.json`(Docker는 `/data/state.json`)을 서버 공유 파일로 사용하므로 서로 다른 브라우저의 저장이 같은 로컬 서버에 반영된다. 파일 저장은 원자적 교체·버전 검사·저장 잠금을 사용하며 최근 50회 상태를 backups/에 보관한다. 이 로컬 파일과 `.env`, `.vercel`, 빌드 산출물은 Git에서 제외한다. 로컬 데이터는 Vercel에 자동 업로드하지 않는다.

```bash
npm test
npm run build
```

브라우저 검증은 저장소 루트에서 `.venv/bin/python labeling/tests/browser_check.py`로 실행한다. 검증은 별도 테스트 서버·`LABEL_LOCAL_STORE` 파일 및 `LABEL_TEST_URL`을 지정해 실제 작업 데이터와 분리한다. QA 작성자 전사는 실제 정답이 아니다.

## 라벨링 순서

1. 이름 입력 → 다음 이미지 맡기.
2. 원본 이미지와 대조해 전사, 특이사항은 메모에 작성.
3. 작성 도중에는 임시 저장. 아직 저장 전인 입력은 브라우저에도 백업된다.
4. 원본 대조를 마친 경우에만 검수 완료.
5. 다른 이미지로 이동하거나 작업 반납. 창을 닫으면 임대는 만료 후 다른 사람이 맡을 수 있다.
6. 평가 담당자는 검수 완료 ZIP을 다운로드해 기존 라벨 검사·병합 도구를 사용한다.

표·화살표 연결 구조 자체를 그리는 화면은 이번 범위가 아니다. 이번 앱은 CER/WER를 위한 텍스트 정답 전사에 집중한다. 표는 행·셀 읽기 순서와 구분을 팀에서 합의하고, 화살표에 붙은 텍스트는 전사한다. 연결·셀 좌표 평가는 별도 구조 라벨을 준비한다.
