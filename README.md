# stt

마이크 입력을 `omnilingual-asr`로 실시간 전사하는 예제입니다.

## 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Linux에서 `sounddevice`, `soundfile` 사용을 위해 필요할 수 있습니다:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libsndfile1
```

## 설치

```bash
pip install -e ".[dev]"
```

## 실행

```bash
stt
```

한국어 인식:

```bash
stt --lang kor_Hang
```

장치 자동 선택:

```bash
STT_DEVICE_NAME="H1essential" stt
```

Windows PowerShell:

```powershell
$env:STT_DEVICE_NAME="H1essential"
stt
```

Windows CMD:

```bat
set STT_DEVICE_NAME=H1essential
stt
```

## 프로젝트 구조

```text
stt/
README.md
pyproject.toml
src/
  stt/
    __init__.py
    __main__.py
    app.py
    cli.py
    service/
      __init__.py
      audio.py
      audio_devices.py
      audio_streaming.py
      transcription.py
tests/
  test_input_channels.py
```

## 아키텍처

- CLI 레이어: `src/stt/cli.py`
- 서비스 레이어: `src/stt/service/`
- 엔트리 포인트: `src/stt/app.py`, `src/stt/__main__.py`

### 흐름

- CLI는 입력 파싱, 장치 선택 UI, 종료 신호 처리만 담당합니다.
- 서비스는 오디오 장치 조회, 스트림 설정 검증, 전사 처리 로직을 담당합니다.
- 엔트리 포인트는 CLI의 `main()`을 호출합니다.
