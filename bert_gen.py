import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from config import get_config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import cleaned_text_to_sequence, extract_bert_feature, extract_bert_feature_onnx
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()
# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


def process_line(x: tuple[str, bool]):
    line, add_blank = x
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(
        phone, tone, Languages[language_str]
    )

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
    safetensors_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.safetensors")

    try:
        # Tenta carregar safetensors primeiro (mais seguro)
        if Path(safetensors_path).exists():
            try:
                from safetensors.torch import load_file
                bert = load_file(safetensors_path)
                # safetensors retorna um dict, pega o tensor
                if isinstance(bert, dict):
                    bert = list(bert.values())[0]
            except ImportError:
                logger.warning("safetensors não instalado, usando torch.load")
                bert = torch.load(bert_path, map_location='cpu', weights_only=True)
        # Tenta carregar .pt com weights_only=True
        elif Path(bert_path).exists():
            bert = torch.load(bert_path, map_location='cpu', weights_only=True)
        else:
            raise FileNotFoundError(f"Arquivo BERT não encontrado: {bert_path}")
        
        assert bert.shape[-1] == len(phone), f"Shape mismatch: {bert.shape[-1]} != {len(phone)}"
    except (FileNotFoundError, AssertionError, RuntimeError) as e:
        # Se não encontrou ou shape não bate, gera novo
        logger.debug(f"Gerando BERT feature para {wav_path}: {str(e)}")
        try:
            bert = extract_bert_feature(text, word2ph, Languages(language_str), device)
        except Exception as bert_error:
            logger.warning(f"Erro ao extrair BERT feature, tentando ONNX: {str(bert_error)}")
            try:
                providers = ["CUDAExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
                bert_np = extract_bert_feature_onnx(text, word2ph, Languages(language_str), providers)
                bert = torch.tensor(bert_np)
            except Exception as onnx_error:
                logger.error(f"Falha ao gerar BERT feature para {wav_path}: {str(onnx_error)}")
                raise
        
        assert bert.shape[-1] == len(phone), f"Generated BERT shape mismatch: {bert.shape[-1]} != {len(phone)}"
        
        # Salva em ambos os formatos
        try:
            torch.save(bert, bert_path)
            try:
                from safetensors.torch import save_file
                save_file({"bert": bert}, safetensors_path)
            except ImportError:
                pass  # safetensors opcional
        except Exception as save_error:
            logger.warning(f"Erro ao salvar BERT feature: {str(save_error)}")


preprocess_text_config = config.preprocess_text_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        # pyopenjtalkの別ワーカー化により、並列処理でエラーがでる模様なので、一旦シングルスレッド強制にする
        num_processes = 1
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(process_line, zip(lines, add_blank)),
                    total=len(lines),
                    file=SAFE_STDOUT,
                    dynamic_ncols=True,
                )
            )

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
