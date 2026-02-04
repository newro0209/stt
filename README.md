# stt

마이크 입력을 `omnilingual-asr`로 실시간 전사하는 간단한 예제입니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Linux에서 `sounddevice`, `soundfile` 사용을 위해 필요할 수 있습니다:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libsndfile1
```

## 실행

```bash
python -m stt
```

한국어 인식:

```bash
python -m stt --lang kor_Hang
```

장치 목록/지정:

```bash
python -m stt --list-devices
python -m stt --device 3
```
