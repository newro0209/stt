import argparse
import os
import queue
import signal
import sys
import termios
import threading
import tty

import numpy as np
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from .service import audio_devices, audio_streaming


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="마이크 입력을 실시간으로 텍스트로 변환합니다.")
    parser.add_argument("--model", default="omniASR_CTC_300M", help="사용할 모델 카드")
    parser.add_argument("--lang", default="eng_Latn", help="언어 코드 (예: eng_Latn, kor_Hang)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="샘플레이트")
    parser.add_argument("--channels", type=int, default=1, help="입력 채널 수")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="전사에 사용할 청크 길이(초)")
    parser.add_argument("--block-seconds", type=float, default=0.5, help="오디오 콜백 블록 길이(초)")
    return parser.parse_args()


def _build_device_menu(devices: list[tuple[int, dict]]) -> list[tuple[str, int | None]]:
    menu: list[tuple[str, int | None]] = []
    hostapis = audio_devices.fetch_hostapis()
    for index, device in devices:
        label = audio_devices.describe_device(device, hostapis, index)
        menu.append((label, index))
    menu.append(("그냥 종료", None))
    return menu


def _render_device_menu(menu: list[tuple[str, int | None]], selected: int) -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    print("위/아래 방향키로 이동하고 Enter로 선택하세요. (q로 종료)")
    for index, (label, _) in enumerate(menu):
        prefix = "▶ " if index == selected else "  "
        print(f"{prefix}{label}")
    sys.stdout.flush()


def _read_key() -> str:
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x1b":
            # 방향키는 ESC 시퀀스로 들어오므로 추가 문자를 읽는다.
            second = sys.stdin.read(1)
            third = sys.stdin.read(1)
            if second == "[" and third == "A":
                return "up"
            if second == "[" and third == "B":
                return "down"
            return "unknown"
        if first in ("\r", "\n"):
            return "enter"
        if first.lower() == "q":
            return "quit"
        return "unknown"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _move_selection(current: int, direction: str, total: int) -> int:
    if direction == "up":
        return (current - 1) % total
    if direction == "down":
        return (current + 1) % total
    return current


def _prompt_device_selection(
    devices: list[tuple[int, dict]],
    key_reader=_read_key,
    input_func=input,
) -> int | None:
    # TTY가 아니면 방향키 입력이 불가해 텍스트 입력 모드로 전환한다.
    if not sys.stdin.isatty():
        return _prompt_device_selection_fallback(devices, input_func=input_func)
    menu = _build_device_menu(devices)
    selected = 0
    while True:
        _render_device_menu(menu, selected)
        key = key_reader()
        if key == "quit":
            return None
        if key == "enter":
            return menu[selected][1]
        selected = _move_selection(selected, key, len(menu))


def _prompt_device_selection_fallback(
    devices: list[tuple[int, dict]],
    input_func=input,
) -> int | None:
    # TTY가 아닐 때도 선택 가능하도록 숫자 입력 흐름을 제공한다.
    available_indices = [index for index, _ in devices]
    for label, _ in _build_device_menu(devices):
        print(label)
    print("TTY 입력이 아니므로 숫자 입력 모드로 전환합니다. 종료는 q 또는 빈 값입니다.")
    while True:
        raw = input_func("사용할 입력 장치 번호를 입력하세요: ").strip()
        if not raw or raw.lower() == "q":
            return None
        if not raw.isdigit():
            print("숫자 형식으로 입력해 주세요.")
            continue
        index = int(raw)
        if index not in available_indices:
            print("목록에 없는 장치 번호입니다. 다시 선택해 주세요.")
            continue
        return index


def _match_device_by_name(
    devices: list[tuple[int, dict]],
    preferred_name: str,
) -> int | None:
    lowered = preferred_name.strip().lower()
    if not lowered:
        return None
    for index, device in devices:
        name = str(device.get("name", "")).lower()
        if lowered in name:
            return index
    return None


def _select_input_device(
    preferred_name: str | None,
    input_func=input,
) -> int | None:
    try:
        devices = audio_devices.list_input_devices()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return None
    if not devices:
        print("사용 가능한 입력 장치가 없습니다.", file=sys.stderr)
        return None
    if preferred_name:
        matched = _match_device_by_name(devices, preferred_name)
        if matched is not None:
            print(f"환경 변수로 지정된 입력 장치 '{preferred_name}'를 선택합니다.")
            return matched
    return _prompt_device_selection(devices, input_func=input_func)


def _install_signal_handlers(
    stop_event: threading.Event,
    reason_box: dict[str, str | None],
) -> None:
    def handle_signal(signum, frame):
        stop_event.set()
        if signum == signal.SIGTSTP:
            reason_box["reason"] = "tstp"
        elif signum == signal.SIGTERM:
            reason_box["reason"] = "term"
        else:
            reason_box["reason"] = "interrupt"
        # 입력 대기 중에도 즉시 빠져나오도록 KeyboardInterrupt를 발생시킨다.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGTSTP, handle_signal)


def _build_audio_runtime() -> tuple[queue.Queue[np.ndarray], threading.Event, dict[str, str | None]]:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    reason_box: dict[str, str | None] = {"reason": None}
    return audio_queue, stop_event, reason_box


def main() -> int:
    args = _parse_args()
    preferred_name = os.environ.get("STT_DEVICE_NAME", "").strip() or None

    try:
        while True:
            selected_device = _select_input_device(preferred_name)
            if selected_device is None:
                return 1

            resolved_channels, error_message, warning_message = audio_devices.resolve_input_channels(
                selected_device,
                args.channels,
            )
            if error_message:
                print(error_message, file=sys.stderr)
                if preferred_name:
                    print("자동 선택 장치가 열리지 않아 수동 선택으로 전환합니다.", file=sys.stderr)
                    preferred_name = None
                continue
            if warning_message:
                print(warning_message, file=sys.stderr)

            resolved_rate, rate_error, rate_warning = audio_devices.resolve_sample_rate(
                selected_device,
                args.sample_rate,
                resolved_channels,
            )
            if rate_error:
                print(rate_error, file=sys.stderr)
                if preferred_name:
                    print("자동 선택 장치가 열리지 않아 수동 선택으로 전환합니다.", file=sys.stderr)
                    preferred_name = None
                continue
            if rate_warning:
                print(rate_warning, file=sys.stderr)

            args.channels = resolved_channels
            args.sample_rate = resolved_rate
            break
    except KeyboardInterrupt:
        print("종료합니다.")
        return 0

    pipeline = ASRInferencePipeline(model_card=args.model)
    audio_queue, stop_event, reason_box = _build_audio_runtime()
    _install_signal_handlers(stop_event, reason_box)

    stream_dtype, dtype_error = audio_devices.resolve_stream_dtype(
        selected_device,
        args.sample_rate,
        args.channels,
    )
    if dtype_error:
        print(dtype_error, file=sys.stderr)
        return 1

    chunk_frames = int(args.sample_rate * args.chunk_seconds)
    block_frames = int(args.sample_rate * args.block_seconds)

    audio_callback = audio_streaming.create_audio_callback(audio_queue)
    worker_thread = audio_streaming.start_worker(
        audio_queue,
        stop_event,
        pipeline,
        args.sample_rate,
        chunk_frames,
        args.channels,
        args.lang,
        stream_dtype,
    )

    try:
        audio_devices.run_stream(
            selected_device,
            args.sample_rate,
            args.channels,
            block_frames,
            audio_callback,
            stop_event,
            stream_dtype,
        )
    except KeyboardInterrupt:
        stop_event.set()
        reason_box["reason"] = "interrupt"
        print("종료합니다.")
    except Exception as exc:
        stop_event.set()
        print(f"오디오 스트림을 열 수 없습니다: {exc}", file=sys.stderr)
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)
        if reason_box["reason"] in {"interrupt", "tstp", "term"}:
            print("종료합니다.")

    return 0
