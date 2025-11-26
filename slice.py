import argparse
import shutil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional

import soundfile as sf
import torch
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


def is_audio_file(file: Path) -> bool:
    supported_extensions = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    return file.suffix.lower() in supported_extensions


def get_stamps(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    min_silence_dur_ms: int = 700,
    min_sec: float = 2,
    max_sec: float = 12,
):
    """
    VAD (Voice Activity Detection) usando Silero.
    """

    (get_speech_timestamps, _, read_audio, *_) = utils
    sampling_rate = 16000  # Silero opera em 8k ou 16k

    min_ms = int(min_sec * 1000)

    wav = read_audio(str(audio_file), sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_s=max_sec,
    )

    return speech_timestamps


def split_wav(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    target_dir: Path,
    min_sec: float = 2,
    max_sec: float = 12,
    min_silence_dur_ms: int = 700,
    time_suffix: bool = False,
) -> tuple[float, int]:

    margin: int = 200  # margem em ms antes e depois de cada trecho detectado
    speech_timestamps = get_stamps(
        vad_model=vad_model,
        utils=utils,
        audio_file=audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )

    data, sr = sf.read(audio_file)
    total_ms = len(data) / sr * 1000

    file_name = audio_file.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    total_time_ms: float = 0
    count = 0

    for i, ts in enumerate(speech_timestamps):
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        if time_suffix:
            file = f"{file_name}-{int(start_ms)}-{int(end_ms)}.wav"
        else:
            file = f"{file_name}-{i}.wav"

        sf.write(str(target_dir / file), segment, sr)
        total_time_ms += end_ms - start_ms
        count += 1

    return total_time_ms / 1000, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_sec", "-m", type=float, default=2)
    parser.add_argument("--max_sec", "-M", type=float, default=12)
    parser.add_argument("--input_dir", "-i", type=str, default="inputs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--min_silence_dur_ms", "-s", type=int, default=700)
    parser.add_argument("--time_suffix", "-t", action="store_true")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=3,
        help="Number of worker threads to use.",
    )
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    model_name = str(args.model_name)
    input_dir = Path(args.input_dir)
    output_dir = dataset_root / model_name / "raw"
    min_sec = args.min_sec
    max_sec = args.max_sec
    min_silence_dur_ms = args.min_silence_dur_ms
    time_suffix = args.time_suffix
    num_processes = args.num_processes

    audio_files = [file for file in input_dir.rglob("*") if is_audio_file(file)]

    logger.info(f"Found {len(audio_files)} audio files.")
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists, deleting...")
        shutil.rmtree(output_dir)

    # Precarrega o modelo Silero (repo oficial)
    _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=True,
        trust_repo=True,
    )

    # Fila de processamento
    def process_queue(
        q: Queue[Optional[Path]],
        result_queue: Queue[tuple[float, int]],
        error_queue: Queue[tuple[Path, Exception]],
    ):
        # Cada worker carrega seu próprio modelo Silero
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )

        while True:
            file = q.get()
            if file is None:
                q.task_done()
                break
            try:
                rel_path = file.relative_to(input_dir)
                time_sec, count = split_wav(
                    vad_model=vad_model,
                    utils=utils,
                    audio_file=file,
                    target_dir=output_dir / rel_path.parent,
                    min_sec=min_sec,
                    max_sec=max_sec,
                    min_silence_dur_ms=min_silence_dur_ms,
                    time_suffix=time_suffix,
                )
                result_queue.put((time_sec, count))
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                error_queue.put((file, e))
                result_queue.put((0, 0))
            finally:
                q.task_done()

    q: Queue[Optional[Path]] = Queue()
    result_queue: Queue[tuple[float, int]] = Queue()
    error_queue: Queue[tuple[Path, Exception]] = Queue()

    num_processes = min(num_processes, len(audio_files))

    threads = [
        Thread(target=process_queue, args=(q, result_queue, error_queue))
        for _ in range(num_processes)
    ]
    for t in threads:
        t.start()

    pbar = tqdm(total=len(audio_files), file=SAFE_STDOUT, dynamic_ncols=True)
    for file in audio_files:
        q.put(file)

    total_sec = 0
    total_count = 0
    for _ in range(len(audio_files)):
        time, count = result_queue.get()
        total_sec += time
        total_count += count
        pbar.update(1)

    q.join()

    for _ in range(num_processes):
        q.put(None)

    for t in threads:
        t.join()

    pbar.close()

    if not error_queue.empty():
        error_str = "Erro ao dividir alguns arquivos:"
        while not error_queue.empty():
            file, e = error_queue.get()
            error_str += f"\n{file}: {e}"
        raise RuntimeError(error_str)

    logger.info(
        f"Slice done! Total time: {total_sec / 60:.2f} min, {total_count} files."
    )
