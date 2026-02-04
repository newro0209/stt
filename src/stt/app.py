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
    # 장치 목록은 사용자에게 선택 근거를 주기 위해 그대로 출력한다.
    devices = sd.query_devices()
    # 각 장치의 핵심 스펙만 노출해 과도한 정보 노이즈를 줄인다.
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
    # 다채널 입력일 수 있으므로 모델 입력을 위해 모노로 합친다.
    if chunk.ndim == 2:
        chunk = np.mean(chunk, axis=1)

    # 모델이 파일 입력을 요구하므로 임시 파일을 생성해 변환한다.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            # 가능한 한 동일한 샘플레이트로 저장해 추가 리샘플링을 피한다.
            sf.write(temp_path, chunk, sample_rate)

        # 배치를 1로 제한해 실시간 처리 지연을 최소화한다.
        transcriptions = pipeline.transcribe([temp_path], lang=[lang], batch_size=1)
        if not transcriptions:
            return ""
        return transcriptions[0]
    finally:
        # 실패 여부와 상관없이 임시 파일을 정리해 디스크 누수를 방지한다.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _resolve_input_channels(
    device_index: int | None,
    requested_channels: int,
) -> tuple[int, str | None, str | None]:
    # 장치 정보를 읽어 채널 범위를 검증하고, 오류/경고 메시지를 분리해 반환한다.
    try:
        device_info = sd.query_devices(device_index, "input")
    except Exception as exc:
        # 장치 조회 실패는 즉시 종료해야 하므로 오류 메시지를 반환한다.
        return requested_channels, f"입력 장치 정보를 가져오지 못했습니다: {exc}", None

    max_inputs = int(device_info.get("max_input_channels", 0))
    if max_inputs <= 0:
        # 입력 채널이 없는 장치는 사용할 수 없으므로 오류 메시지를 반환한다.
        return requested_channels, "선택한 입력 장치가 입력 채널을 제공하지 않습니다.", None

    if requested_channels > max_inputs:
        # 사용자가 요청한 채널 수를 장치 제한에 맞게 하향 조정한다.
        warning = (
            f"요청한 채널 수({requested_channels})가 장치 최대 입력 채널({max_inputs})을 "
            f"초과합니다. {max_inputs}로 조정합니다."
        )
        return max_inputs, None, warning

    # 정상 범위라면 요청 값을 그대로 사용한다.
    return requested_channels, None, None


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
        # 장치 목록만 요청된 경우, 더 이상의 초기화를 하지 않는다.
        _print_devices()
        return 0

    # 장치의 입력 채널 수를 검증해 스트림 생성 오류를 사전에 방지한다.
    resolved_channels, error_message, warning_message = _resolve_input_channels(
        args.device,
        args.channels,
    )
    if error_message:
        # 사용자에게 실패 이유와 해결 경로를 동시에 안내한다.
        print(error_message, file=sys.stderr)
        print("사용 가능한 장치를 보려면 --list-devices 를 사용하세요.", file=sys.stderr)
        return 1
    if warning_message:
        # 치명적이지 않은 경우는 경고만 출력하고 계속 진행한다.
        print(warning_message, file=sys.stderr)
    args.channels = resolved_channels

    pipeline = ASRInferencePipeline(model_card=args.model)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    chunk_frames = int(args.sample_rate * args.chunk_seconds)
    block_frames = int(args.sample_rate * args.block_seconds)

    def audio_callback(indata, frames, time_info, status):
        # 콜백 경고는 실시간 문제 진단에 도움되므로 stderr로 전달한다.
        if status:
            print(f"오디오 상태 경고: {status}", file=sys.stderr)
        # 큐에 복사본을 넣어 콜백 이후 버퍼가 변형되는 문제를 방지한다.
        audio_queue.put(indata.copy())

    def worker():
        # 채널 수를 유지한 버퍼로 누적해 청크 단위 전사를 수행한다.
        buffer = np.zeros((0, args.channels), dtype=np.float32)
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 누적 버퍼를 늘리고, 충분한 길이만큼 잘라 전사를 반복한다.
            buffer = np.concatenate([buffer, data], axis=0)
            while buffer.shape[0] >= chunk_frames:
                chunk = buffer[:chunk_frames]
                buffer = buffer[chunk_frames:]

                # 청크별로 처리 시간을 측정해 체감 지연을 파악한다.
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
