# 서버 관리자에게 전달할 배포 안내

**이 폴더만 전달하면 됩니다.** 122장의 이미지·초기 정답·웹 화면·공동 저장 서버가 모두 포함되어 있습니다. Docker Compose가 설치된 서버에서 실행하며 GPU·Python·GitHub 토큰·별도 DB는 필요 없습니다. 사용자는 이름만 입력합니다.

## A. 새 도메인 또는 서브도메인으로 배포

1. 압축을 풀고 이 파일이 있는 폴더로 이동합니다.
2. `labels.example.com`의 DNS A 레코드를 서버 공인 IP로 연결합니다. AAAA를 설정했다면 같은 서버의 정상 IPv6 주소를 사용합니다. 서버의 80/443 포트를 연결할 수 있어야 합니다.
3. 아래 명령을 실행합니다.

```bash
cp .env.selfhost.example .env
```

`.env`에서 `LABEL_DOMAIN=labels.example.com`을 실제 도메인으로 바꿉니다. `https://`와 경로를 붙이지 않고 호스트 이름만 입력합니다.

```bash
docker compose -f compose.yaml -f compose.domain.yaml up -d --build
```

이후 **https://설정한도메인**에 접속합니다. Caddy가 도메인 HTTPS 인증서 발급·갱신을 관리합니다. DNS·포트가 준비되어야 인증서가 발급됩니다. 첫 실행에는 Docker 이미지 다운로드·빌드 시간이 필요합니다.

## B. 이미 Nginx/Caddy 등이 80/443을 사용 중인 서버

앱만 띄우고 기존 웹서버에 서브도메인 프록시를 추가합니다.

```bash
docker compose up -d --build
```

앱은 서버 내부 `http://127.0.0.1:8080`에서 실행됩니다. 도메인 관리자에게 이 주소로 reverse proxy를 연결해 달라고 요청합니다. 이 경우 `compose.domain.yaml`은 실행하지 않습니다.

Nginx 설정 예시(기존 HTTPS 설정 안의 location 또는 해당 서브도메인 서버 블록):

```nginx
server {
    listen 80;
    server_name labels.example.com;
    client_max_body_size 256k;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

HTTPS는 기존 도메인 관리 방식으로 설정합니다. 독립 서브도메인 루트(`/`)로 연결하는 구성이며 `/labeling` 같은 경로 아래 설치는 별도 URL 설정이 필요합니다. 8080이 사용 중이면 `.env`에 `LABEL_HTTP_PORT=다른포트`를 적고 proxy 주소도 같은 포트로 바꿉니다.

## 실행·확인·업데이트

```bash
# 앱 상태 확인
curl http://127.0.0.1:8080/healthz

# 로그 확인
docker compose logs --tail=100 app

# 파일 업데이트 후 앱 재빌드
docker compose up -d --build app
```

도메인까지 실행한 서비스는 관리 명령에도 `-f compose.yaml -f compose.domain.yaml`을 사용합니다. 다른 폴더로 이동하거나 Compose 프로젝트 이름을 바꾸면 별도 데이터 볼륨이 만들어질 수 있으므로 기존 프로젝트를 유지합니다.

## 정답 저장 위치와 백업

서버 저장은 `labeling_data` Docker 볼륨의 `/data/state.json`입니다. 컨테이너를 재시작·재생성해도 유지됩니다. 저장 직전 상태는 `/data/backups/`에 최근 50회까지 자동 보관합니다. 이는 50일 보관이 아니라 **최근 50회 변경**입니다. `docker compose down -v`는 데이터 볼륨까지 삭제하므로 사용하지 않습니다.

웹의 **정답 파일 다운로드**에서 개별 JSON ZIP 또는 통합 JSON을 내려받을 수 있습니다. 전체 상태 백업(이력·revision 포함)은 다음 명령으로 서버 밖에도 보관합니다.

```bash
docker compose exec -T app node tools/state.mjs backup > labeling-backup.json
```

백업을 새 서버에 복원할 때도 같은 초기 이미지·정답 패키지를 사용합니다. 복원 전 앱을 멈추고 실행합니다.

```bash
docker compose stop app
docker compose run --rm -T --no-deps app node tools/state.mjs restore --replace < labeling-backup.json
docker compose up -d app
```

이 명령은 전사·검수 상태·이력을 복구하고 이전 브라우저의 작업 임대는 해제합니다. 교체 전 파일은 추가 백업으로 남깁니다. 일반 JSON/ZIP 내보내기와 이 전체 상태 백업은 용도가 다릅니다.

## 운영 방식

- 기본 상태: 기존 검수 완료 5장 보호, 추가 117장 미작성.
- 이름 입력 → 작업 맡기 → 원문 전사·메모 → 임시 저장 → 원본 대조 후 검수 완료.
- 작성자 이름은 로그인 인증이 아닙니다. 공유 주소에 접근한 사람은 추가 라벨을 작성할 수 있습니다.
- 이 구성은 한 서버의 영구 디스크와 **앱 인스턴스 1개**를 기준으로 합니다.
- 데이터 볼륨·작업 파일·브라우저 검증 데이터가 포함되지 않은 초기 배포 패키지입니다.
- 서버가 아닌 정적 파일 전용 호스팅은 공동 저장 서버를 실행할 수 없습니다. Vercel을 쓰는 경우 `README.md`의 별도 Vercel 배포 설정을 사용합니다.

Node.js 22 서버를 직접 운영할 수도 있습니다. `LABEL_LOCAL_STORE`를 영구 디스크 경로로 설정하고 `HOST=127.0.0.1 PORT=8080 npm start`로 실행한 뒤 기존 프록시에 연결합니다.
