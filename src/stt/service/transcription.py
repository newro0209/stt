import os
import tempfile

import numpy as np
import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from .audio import normalize_audio, split_channels


def transcribe_channel(
    pipeline: ASRInferencePipeline,
    channel_chunk: np.ndarray,
    sample_rate: int,
    lang: str,
) -> str:
    temp_path = None
    try:
        # 파이프라인이 파일 입력을 요구하므로 임시 파일로 변환한다.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, channel_chunk, sample_rate)

        # 배치를 1로 고정해 실시간 지연을 최소화한다.
        transcriptions = pipeline.transcribe([temp_path], lang=[lang], batch_size=1)
        if not transcriptions:
            return ""
        return transcriptions[0]
    finally:
        # 실패 여부와 무관하게 임시 파일을 정리해 누수를 막는다.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def transcribe_chunk(
    pipeline: ASRInferencePipeline,
    chunk: np.ndarray,
    sample_rate: int,
    lang: str,
    channels: int,
    dtype: str,
) -> str:
    # 채널별로 전사한 뒤 하나의 문자열로 합친다.
    normalized = normalize_audio(chunk, dtype)
    channel_chunks = split_channels(normalized, channels)
    results: list[str] = []
    for index, channel_chunk in enumerate(channel_chunks, start=1):
        text = transcribe_channel(pipeline, channel_chunk, sample_rate, lang)
        results.append(f"CH{index}: {text}")
    return " | ".join(results)
