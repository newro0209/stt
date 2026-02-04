import time


def _get_sounddevice():
    try:
        # PortAudio 초기화 실패를 RuntimeError로 통일한다.
        import sounddevice as sd  # pylint: disable=import-error
    except Exception as exc:
        raise RuntimeError(f"오디오 모듈을 불러올 수 없습니다: {exc}") from exc
    return sd


def fetch_devices() -> list[dict]:
    try:
        sd = _get_sounddevice()
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"오디오 장치 목록을 가져올 수 없습니다: {exc}") from exc
    return list(devices)


def fetch_hostapis() -> list[dict]:
    try:
        sd = _get_sounddevice()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise RuntimeError(f"오디오 호스트 API를 가져올 수 없습니다: {exc}") from exc
    return list(hostapis)


def list_input_devices() -> list[tuple[int, dict]]:
    devices = fetch_devices()
    input_devices: list[tuple[int, dict]] = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            input_devices.append((index, device))
    return input_devices


def describe_device(device: dict, hostapis: list[dict], index: int) -> str:
    hostapi_name = hostapis[int(device.get("hostapi", 0))].get("name", "")
    name = device.get("name", "")
    max_inputs = device.get("max_input_channels", 0)
    default_rate = device.get("default_samplerate", 0)
    return (
        f"[{index}] {name} ({hostapi_name}) | 입력 채널: {max_inputs} | "
        f"기본 샘플레이트: {default_rate}"
    )


def resolve_input_channels(
    device_index: int | None,
    requested_channels: int,
) -> tuple[int, str | None, str | None]:
    try:
        sd = _get_sounddevice()
        device_info = sd.query_devices(device_index, "input")
    except Exception as exc:
        return requested_channels, f"입력 장치 정보를 가져오지 못했습니다: {exc}", None

    max_inputs = int(device_info.get("max_input_channels", 0))
    if max_inputs <= 0:
        return requested_channels, "선택한 입력 장치가 입력 채널을 제공하지 않습니다.", None

    if requested_channels > max_inputs:
        warning = (
            f"요청한 채널 수({requested_channels})가 장치 최대 입력 채널({max_inputs})을 "
            f"초과합니다. {max_inputs}로 조정합니다."
        )
        return max_inputs, None, warning

    return requested_channels, None, None


def resolve_sample_rate(
    device_index: int,
    requested_sample_rate: int,
    channels: int,
) -> tuple[int, str | None, str | None]:
    try:
        sd = _get_sounddevice()
        sd.check_input_settings(
            device=device_index,
            samplerate=requested_sample_rate,
            channels=channels,
        )
        return requested_sample_rate, None, None
    except Exception as exc:
        primary_error = str(exc)

    sd = _get_sounddevice()
    device_info = sd.query_devices(device_index, "input")
    fallback_rate = int(device_info.get("default_samplerate", 0))
    if fallback_rate > 0 and fallback_rate != requested_sample_rate:
        try:
            # 기본 샘플레이트가 유효하면 폴백으로 사용한다.
            sd.check_input_settings(
                device=device_index,
                samplerate=fallback_rate,
                channels=channels,
            )
            warning = (
                f"요청한 샘플레이트({requested_sample_rate})가 지원되지 않아 "
                f"기본 샘플레이트({fallback_rate})로 조정합니다."
            )
            return fallback_rate, None, warning
        except Exception as exc:
            return requested_sample_rate, f"선택한 장치 설정을 열 수 없습니다: {exc}", None

    return requested_sample_rate, f"선택한 장치 설정을 열 수 없습니다: {primary_error}", None


def resolve_stream_dtype(
    device_index: int,
    sample_rate: int,
    channels: int,
) -> tuple[str, str | None]:
    # 지원 dtype을 순회해 스트림 생성 실패를 피한다.
    for candidate in ("float32", "int16"):
        try:
            sd = _get_sounddevice()
            sd.check_input_settings(
                device=device_index,
                samplerate=sample_rate,
                channels=channels,
                dtype=candidate,
            )
            return candidate, None
        except Exception as exc:
            last_error = str(exc)
    return "float32", f"지원되는 오디오 포맷을 찾지 못했습니다: {last_error}"


def run_stream(
    device_index: int,
    sample_rate: int,
    channels: int,
    block_frames: int,
    audio_callback,
    stop_event,
    dtype: str,
) -> None:
    sd = _get_sounddevice()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        blocksize=block_frames,
        device=device_index,
        callback=audio_callback,
    ):
        print("마이크 입력을 듣는 중입니다. 종료하려면 Ctrl+C를 누르세요.")
        while not stop_event.is_set():
            time.sleep(0.2)
