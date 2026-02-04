import queue
import sys
import threading
import time

import numpy as np
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from .audio import normalize_audio
from .transcription import transcribe_chunk


def create_audio_callback(audio_queue: queue.Queue[np.ndarray]):
    def audio_callback(indata, frames, time_info, status):
        # 콜백 버퍼는 외부에서 변형될 수 있어 복사본을 큐에 넣는다.
        if status:
            print(f"오디오 상태 경고: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    return audio_callback


def start_worker(
    audio_queue: queue.Queue[np.ndarray],
    stop_event: threading.Event,
    pipeline: ASRInferencePipeline,
    sample_rate: int,
    chunk_frames: int,
    channels: int,
    lang: str,
    dtype: str,
) -> threading.Thread:
    def worker():
        # 큐에서 들어온 오디오를 누적해 고정 길이 청크로 전사한다.
        buffer = np.zeros((0, channels), dtype=np.float32)
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            data = normalize_audio(data, dtype)
            buffer = np.concatenate([buffer, data], axis=0)
            while buffer.shape[0] >= chunk_frames:
                chunk = buffer[:chunk_frames]
                buffer = buffer[chunk_frames:]

                # 처리 시간을 측정해 지연 여부를 확인한다.
                started = time.time()
                text = transcribe_chunk(
                    pipeline,
                    chunk,
                    sample_rate,
                    lang,
                    channels,
                    dtype,
                )
                elapsed = time.time() - started

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {text} (소요 {elapsed:.2f}s)")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    return worker_thread
