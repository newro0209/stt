import argparse
import os
import queue
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def _print_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        name = device.get("name", "")
        max_inputs = device.get("max_input_channels", 0)
        default_rate = device.get("default_samplerate", 0)
        print(f"[{index}] {name} | 입력 채널: {max_inputs} | 기본 샘플레이트: {default_rate}")


def _transcribe_chunk(
    pipeline: ASRInferencePipeline,
    chunk: np.ndarray,
    sample_rate: int,
    lang: str,
) -> str:
    if chunk.ndim == 2:
        chunk = np.mean(chunk, axis=1)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, chunk, sample_rate)

        transcriptions = pipeline.transcribe([temp_path], lang=[lang], batch_size=1)
        if not transcriptions:
            return ""
        return transcriptions[0]
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="마이크 입력을 실시간으로 텍스트로 변환합니다.")
    parser.add_argument("--model", default="omniASR_CTC_300M", help="사용할 모델 카드")
    parser.add_argument("--lang", default="eng_Latn", help="언어 코드 (예: eng_Latn, kor_Hang)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="샘플레이트")
    parser.add_argument("--channels", type=int, default=1, help="입력 채널 수")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="전사에 사용할 청크 길이(초)")
    parser.add_argument("--block-seconds", type=float, default=0.5, help="오디오 콜백 블록 길이(초)")
    parser.add_argument("--device", type=int, default=None, help="입력 장치 인덱스")
    parser.add_argument("--list-devices", action="store_true", help="입력 장치 목록 출력")

    args = parser.parse_args()

    if args.list_devices:
        _print_devices()
        return 0

    pipeline = ASRInferencePipeline(model_card=args.model)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    chunk_frames = int(args.sample_rate * args.chunk_seconds)
    block_frames = int(args.sample_rate * args.block_seconds)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"오디오 상태 경고: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    def worker():
        buffer = np.zeros((0, args.channels), dtype=np.float32)
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, data], axis=0)
            while buffer.shape[0] >= chunk_frames:
                chunk = buffer[:chunk_frames]
                buffer = buffer[chunk_frames:]

                started = time.time()
                text = _transcribe_chunk(pipeline, chunk, args.sample_rate, args.lang)
                elapsed = time.time() - started

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {text} (소요 {elapsed:.2f}s)")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=args.channels,
            dtype="float32",
            blocksize=block_frames,
            device=args.device,
            callback=audio_callback,
        ):
            print("마이크 입력을 듣는 중입니다. 종료하려면 Ctrl+C를 누르세요.")
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("종료합니다.")
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
