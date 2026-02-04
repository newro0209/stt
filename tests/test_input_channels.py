import sounddevice as sd

from stt import app


def test_resolve_input_channels_adjusts_to_device_limit(monkeypatch):
    # 장치가 1채널만 지원하는 상황을 재현한다.
    def fake_query_devices(device_index, kind):
        return {"max_input_channels": 1}

    # 외부 장치 조회를 모킹해 테스트를 안정화한다.
    monkeypatch.setattr(sd, "query_devices", fake_query_devices)

    # 요청이 초과인 경우 채널이 하향 조정되는지 확인한다.
    channels, error_message, warning_message = app._resolve_input_channels(0, 2)

    assert channels == 1
    assert error_message is None
    assert warning_message is not None


def test_resolve_input_channels_ok_when_within_limit(monkeypatch):
    # 장치가 2채널을 지원하는 상황을 재현한다.
    def fake_query_devices(device_index, kind):
        return {"max_input_channels": 2}

    # 외부 장치 조회를 모킹해 테스트를 안정화한다.
    monkeypatch.setattr(sd, "query_devices", fake_query_devices)

    # 정상 범위에서는 경고나 오류 없이 그대로 유지되어야 한다.
    channels, error_message, warning_message = app._resolve_input_channels(0, 1)

    assert channels == 1
    assert error_message is None
    assert warning_message is None


def test_resolve_input_channels_fails_when_no_input_channels(monkeypatch):
    # 입력 채널이 없는 장치를 재현한다.
    def fake_query_devices(device_index, kind):
        return {"max_input_channels": 0}

    # 외부 장치 조회를 모킹해 테스트를 안정화한다.
    monkeypatch.setattr(sd, "query_devices", fake_query_devices)

    # 입력 채널이 없으면 오류 메시지를 반환해야 한다.
    channels, error_message, warning_message = app._resolve_input_channels(0, 1)

    assert channels == 1
    assert error_message is not None
    assert warning_message is None


def test_resolve_input_channels_fails_on_query_error(monkeypatch):
    # 장치 조회 자체가 실패하는 상황을 재현한다.
    def fake_query_devices(device_index, kind):
        raise RuntimeError("boom")

    # 외부 장치 조회를 모킹해 테스트를 안정화한다.
    monkeypatch.setattr(sd, "query_devices", fake_query_devices)

    # 조회 실패 시 오류 메시지를 반환해야 한다.
    channels, error_message, warning_message = app._resolve_input_channels(0, 1)

    assert channels == 1
    assert error_message is not None
    assert warning_message is None
