# lora-simulator

간단한 LoRa 네트워크 시뮬레이터 저장소입니다. 로컬에서 시뮬레이션을 실행하고 실험 결과를 수집하도록 구성되어 있습니다.

빠른 시작

- 파이썬 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # 필요 시 requirements.txt 추가
```

- 시뮬레이터 실행 예

```bash
python app.py
```

무엇을 깃에 올릴까

- 포함할 것:
  - 소스 코드 (`app.py`, `example_usage.py`, `env/`, `agents/`, `baselines/`, `utils/`, `experiments/`, `docs/`)
  - 작은 설정 파일 및 스크립트
  - `README.md`, `requirements.txt` (가능하면)

- 제외할 것 (.gitignore에 추가됨):
  - 가상환경 디렉터리 (`venv/`, `.venv/` 등)
  - 대용량 출력 및 결과 (`outputs/`)
  - 에디터 설정 (`.vscode/`, `.idea/`)
  - 임시/캐시 파일 (`__pycache__/`, `*.pyc` 등)
  - 민감 정보 (`.env`)

GitHub에 올리는 방법 (추천)

1. 로컬에서 Git 초기화 및 커밋

```bash
git init
git add .
git commit -m "Initial commit: lora-simulator"
```

2. GitHub에서 리포지터리 생성 (웹 또는 `gh` CLI)

웹: https://github.com/new 에서 `lora-simulator`로 만들기 (Public)

또는 `gh`가 설치되어 있고 로그인되어 있다면:

```bash
gh repo create kimbongdda/lora-simulator --public --source=. --remote=origin --push
```

3. 수동으로 원격 추가 후 푸시

```bash
git remote add origin git@github.com:kimbongdda/lora-simulator.git
git branch -M main
git push -u origin main
```

주의

- `outputs/` 같은 대용량 파일은 Git LFS 사용을 권장하거나 리포지터리에 포함하지 마세요.
- 인증 문제는 SSH 키 설정 또는 `gh auth login`으로 해결하세요.
