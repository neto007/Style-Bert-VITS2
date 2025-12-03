import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import gradio as gr
import yaml

from config import get_path_config
from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_bert_vits2.utils.subprocess import run_script_with_log, second_elem_of


logger_handler = None
tensorboard_executed = False

path_config = get_path_config()
dataset_root = path_config.dataset_root


@dataclass
class PathsForPreprocess:
    dataset_path: Path
    esd_path: Path
    train_path: Path
    val_path: Path
    config_path: Path


def get_path(model_name: str) -> PathsForPreprocess:
    assert model_name != "", "O nome do modelo não pode estar vazio"
    dataset_path = dataset_root / model_name
    esd_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"
    return PathsForPreprocess(dataset_path, esd_path, train_path, val_path, config_path)


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # Salvar logs de pré-processamento em arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}"
    )

    default_config_path = (
        "configs/config.json" if not use_jp_extra else "configs/config_jp_extra.json"
    )

    with open(default_config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(paths.train_path)
    config["data"]["validation_files"] = str(paths.val_path)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_EN_bert"] = freeze_EN_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False  # Deve ser False por padrão, mas por precaução

    # Atualmente é padrão, mas antes não estava na versão não JP-Extra e causava bugs, então por precaução
    config["data"]["use_jp_extra"] = use_jp_extra

    model_path = paths.dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
        )
        shutil.copytree(
            src=model_path,
            dst=paths.dataset_path / "models_backup",
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    pretrained_dir = Path("pretrained" if not use_jp_extra else "pretrained_jp_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Passo 1: Pasta {pretrained_dir} não encontrada.")
        return f"Erro no Passo 1: Pasta {pretrained_dir} não encontrada."

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not Path("config.yml").exists():
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Passo 1: inicialização concluída.")
    return "Passo 1: A inicialização foi concluída."


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Passo 2: iniciando reamostragem...")
    dataset_path = get_path(model_name).dataset_path
    input_dir = dataset_path / "raw"
    output_dir = dataset_path / "wavs"
    cmd = [
        "resample.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Passo 2: reamostragem falhou.")
        return f"Erro no Passo 2: Falha no pré-processamento do arquivo de áudio:\n{message}"
    elif message:
        logger.warning("Passo 2: reamostragem concluída com stderr.")
        return f"Passo 2: O pré-processamento do arquivo de áudio foi concluído com avisos:\n{message}"
    logger.success("Passo 2: reamostragem concluída.")
    return "Passo 2: O pré-processamento do arquivo de áudio foi concluído."


def preprocess_text(
    model_name: str, use_jp_extra: bool, val_per_spk: int, max_val_total: int, clean: bool
):
    logger.info("Passo 3: iniciando pré-processamento de texto...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Passo 3: {paths.esd_path} não encontrado.")
        return f"Erro no Passo 3: O arquivo de transcrição {paths.esd_path} não foi encontrado."

    cmd = [
        "preprocess_text.py",
        "--metadata",
        str(paths.esd_path),
        "--train_config",
        str(paths.config_path),
        "--val_per_spk",
        str(val_per_spk),
        "--val_max",
        str(max_val_total),
    ]
    if clean:
        cmd.append("--clean")
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Passo 3: pré-processamento de texto falhou.")
        return f"Erro no Passo 3: Falha no pré-processamento do arquivo de transcrição:\n{message}"
    elif message:
        logger.warning("Passo 3: pré-processamento de texto concluído com stderr.")
        return f"Passo 3: O pré-processamento do arquivo de transcrição foi concluído com avisos:\n{message}"
    logger.success("Passo 3: pré-processamento de texto concluído.")
    return "Passo 3: O pré-processamento do arquivo de transcrição foi concluído."


def bert_gen(model_name: str, num_processes: int):
    logger.info("Passo 4: iniciando bert_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path), "--num_processes", str(num_processes)]
    )
    if not success:
        logger.error("Passo 4: bert_gen falhou.")
        return f"Erro no Passo 4: Falha na geração do arquivo de características BERT:\n{message}"
    elif message:
        logger.warning("Passo 4: bert_gen concluído com stderr.")
        return f"Passo 4: A geração do arquivo de características BERT foi concluída com avisos:\n{message}"
    logger.success("Passo 4: bert_gen concluído.")
    return "Passo 4: A geração do arquivo de características BERT foi concluída."


def style_gen(model_name: str, num_processes: int):
    logger.info("Passo 5: iniciando style_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            str(config_path),
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error("Passo 5: style_gen falhou.")
        return f"Erro no Passo 5: Falha na geração do arquivo de características de estilo:\n{message}"
    elif message:
        logger.warning("Passo 5: style_gen concluído com stderr.")
        return f"Passo 5: A geração do arquivo de características de estilo foi concluída com avisos:\n{message}"
    logger.success("Passo 5: style_gen concluído.")
    return "Passo 5: A geração do arquivo de características de estilo foi concluída."


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    val_per_spk: int,
    max_val_total: int,
    log_interval: int,
    clean: bool,
    bf16_run: bool,
    fp16_run: bool,
):
    if model_name == "":
        return "Erro: Por favor, insira o nome do modelo"

    # Verificar se config.json existe
    if not (dataset_root / model_name / "config.json").exists():
        return "Erro: O arquivo de configuração não foi encontrado. Por favor, execute o pré-processamento primeiro."

    message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        log_interval=log_interval,
        bf16_run=bf16_run,
        fp16_run=fp16_run,
    )
    if "Erro" in message:
        return message
    message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if "Erro" in message:
        return message

    message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        val_per_spk=val_per_spk,
        max_val_total=max_val_total,
        clean=clean,
    )
    if "Erro" in message:
        return message
    message = bert_gen(
        model_name=model_name, num_processes=num_processes
    )  # bert_gen é pesado, não altere o número de processos
    if "Erro" in message:
        return message
    message = style_gen(model_name=model_name, num_processes=num_processes)
    if "Erro" in message:
        return message
    logger.success("Sucesso: Todo o pré-processamento concluído!")
    return (
        "Sucesso: Todo o pré-processamento foi concluído. Recomenda-se verificar o terminal para quaisquer anomalias."
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    # Atualizar config.yml para o caso de retomar o treinamento
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    train_py = "train_ms.py" if not use_jp_extra else "train_ms_jp_extra.py"
    cmd = [
        train_py,
        "--config",
        str(paths.config_path),
        "--model",
        str(paths.dataset_path),
    ]
    if skip_style:
        cmd.append("--skip_default_style")
    if speedup:
        cmd.append("--speedup")
    if not_use_custom_batch_sampler:
        cmd.append("--not_use_custom_batch_sampler")
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        logger.error("Treinamento falhou.")
        return f"Erro: Falha no treinamento:\n{message}"
    elif message:
        logger.warning("Treinamento concluído com stderr.")
        return f"Sucesso: O treinamento foi concluído com avisos:\n{message}"
    logger.success("Treinamento concluído.")
    return "O treinamento foi concluído."


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True  # Se a porta estiver aberta
        except OSError:
            pass  # Se a porta ainda não estiver aberta

        if time.time() - start_time > timeout:
            return False  # Tempo limite

        time.sleep(0.1)


def run_tensorboard(model_name: str):
    global tensorboard_executed
    if not tensorboard_executed:
        python = sys.executable
        tensorboard_cmd = [
            python,
            "-m",
            "tensorboard.main",
            "--logdir",
            f"Data/{model_name}/models",
        ]
        subprocess.Popen(
            tensorboard_cmd,
            stdout=SAFE_STDOUT,  # type: ignore
            stderr=SAFE_STDOUT,  # type: ignore
        )
        yield gr.Button("Iniciando…")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard não iniciou no tempo esperado.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Abrir Tensorboard")


change_log_md = """
## Mudanças na Ver 2.5
- O modelo padrão foi alterado para `koharune-ami` e `amitaro`.
- O método de pré-processamento foi aprimorado para ser mais rápido e eficiente.
- Adicionado suporte para treinamento com `uv` para maior velocidade.
"""

how_to_md = """
## Como usar

1. **Preparação de Dados**: Coloque os arquivos de áudio e transcrição na pasta `Data`.
2. **Pré-processamento**: Execute os passos 1 a 5 para preparar os dados para treinamento.
3. **Treinamento**: Ajuste os hiperparâmetros e inicie o treinamento.

### JP-Extra
Se você ativar a opção `Usar versão JP-Extra`, o modelo será otimizado para japonês, resultando em melhor qualidade de voz, mas perderá a capacidade de falar inglês e chinês.
"""

prepare_md = """
## Preparação de Dados

Certifique-se de que seus dados estejam organizados da seguinte maneira:
```
Data
├── seu_modelo
│   ├── raw
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   └── esd.list
```
O arquivo `esd.list` deve conter o caminho do arquivo de áudio, o nome do falante, o idioma e o texto da transcrição, separados por `|`.
Exemplo:
`raw/audio1.wav|seu_modelo|JP|Olá, como vai?`
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("Como usar", open=False):
            gr.Markdown(how_to_md)
        with gr.Accordion("Preparação de Dados", open=False):
            gr.Markdown(prepare_md)

        model_name = gr.Textbox(label="Nome do Modelo")

        with gr.Tab("Pré-processamento Automático"):
            gr.Markdown("Execute todos os passos de pré-processamento automaticamente.")
            with gr.Row(variant="panel"):
                with gr.Column():
                    use_jp_extra = gr.Checkbox(
                        label="Usar versão JP-Extra (Melhora o desempenho em japonês, mas não fala inglês e chinês)",
                        value=True,
                    )
                    batch_size = gr.Slider(
                        label="Tamanho do Lote",
                        info="Se a velocidade de treinamento for lenta, diminua. Se tiver VRAM de sobra, aumente. Estimativa de VRAM para JP-Extra: 1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB",
                        value=2,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    epochs = gr.Slider(
                        label="Número de Épocas",
                        info="100 geralmente é suficiente, mas mais pode melhorar a qualidade",
                        value=100,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    save_every_steps = gr.Slider(
                        label="Salvar resultados a cada N passos",
                        info="Note que isso é diferente do número de épocas",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                    )
                    normalize = gr.Checkbox(
                        label="Normalizar volume do áudio (se o volume não for consistente)",
                        value=False,
                    )
                    trim = gr.Checkbox(
                        label="Remover silêncio no início e no fim do áudio",
                        value=False,
                    )
                    clean = gr.Checkbox(
                        label="Ignorar arquivos ilegíveis (se desmarcado, interrompe com erro)",
                        value=False,
                    )
                    with gr.Accordion("Configurações Avançadas", open=False):
                        num_processes = gr.Slider(
                            label="Número de Processos",
                            info="Número de processos paralelos durante o pré-processamento. Reduza se congelar.",
                            value=cpu_count() // 2,
                            minimum=1,
                            maximum=cpu_count(),
                            step=1,
                        )
                        val_per_spk = gr.Slider(
                            label="Dados de Validação por Falante",
                            info="Não usado para treinamento, mas para comparar áudio original e sintetizado no tensorboard",
                            value=0,
                            minimum=0,
                            maximum=100,
                            step=1,
                        )
                        max_val_total = gr.Slider(
                            label="Máximo Total de Dados de Validação",
                            value=8,
                            minimum=0,
                            maximum=100,
                            step=1,
                        )
                        log_interval = gr.Slider(
                            label="Intervalo de Log do Tensorboard",
                            info="Se quiser ver detalhes no Tensorboard, diminua este valor",
                            value=200,
                            minimum=10,
                            maximum=1000,
                            step=10,
                        )
                        bf16_run = gr.Checkbox(
                            label="Usar bf16 (Requer GPU Ampere ou mais recente)",
                            value=False,
                        )
                        fp16_run = gr.Checkbox(
                            label="Usar fp16",
                            value=True,
                        )
                        gr.Markdown("Congelar partes específicas durante o treinamento")
                        freeze_EN_bert = gr.Checkbox(
                            label="Congelar BERT em inglês",
                            value=False,
                        )
                        freeze_JP_bert = gr.Checkbox(
                            label="Congelar BERT em japonês",
                            value=False,
                        )
                        freeze_ZH_bert = gr.Checkbox(
                            label="Congelar BERT em chinês",
                            value=False,
                        )
                        freeze_style = gr.Checkbox(
                            label="Congelar parte de estilo",
                            value=False,
                        )
                        freeze_decoder = gr.Checkbox(
                            label="Congelar parte do decodificador",
                            value=False,
                        )

                with gr.Column():
                    preprocess_button = gr.Button(
                        value="Executar Pré-processamento Automático", variant="primary"
                    )
                    info_all = gr.Textbox(label="Status")
        with gr.Tab("Pré-processamento Manual"):
            gr.Markdown("Execute cada passo do pré-processamento manualmente.")

            with gr.Group():
                gr.Markdown("### Passo 1: Geração do Arquivo de Configuração")
                use_jp_extra_manual = gr.Checkbox(
                    label="Usar versão JP-Extra",
                    value=True,
                )
                batch_size_manual = gr.Slider(
                    label="Tamanho do Lote",
                    value=2,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs_manual = gr.Slider(
                    label="Número de Épocas",
                    value=100,
                    minimum=1,
                    maximum=1000,
                    step=1,
                )
                save_every_steps_manual = gr.Slider(
                    label="Salvar resultados a cada N passos",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                log_interval_manual = gr.Slider(
                    label="Intervalo de Log do Tensorboard",
                    value=200,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                bf16_run_manual = gr.Checkbox(
                    label="Usar bf16 (Requer GPU Ampere ou mais recente)",
                    value=False,
                )
                fp16_run_manual = gr.Checkbox(
                    label="Usar fp16",
                    value=True,
                )
                freeze_EN_bert_manual = gr.Checkbox(
                    label="Congelar BERT em inglês",
                    value=False,
                )
                freeze_JP_bert_manual = gr.Checkbox(
                    label="Congelar BERT em japonês",
                    value=False,
                )
                freeze_ZH_bert_manual = gr.Checkbox(
                    label="Congelar BERT em chinês",
                    value=False,
                )
                freeze_style_manual = gr.Checkbox(
                    label="Congelar parte de estilo",
                    value=False,
                )
                freeze_decoder_manual = gr.Checkbox(
                    label="Congelar parte do decodificador",
                    value=False,
                )
                generate_config_btn = gr.Button(value="Executar", variant="primary")
                info_init = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("### Passo 2: Pré-processamento de Áudio")
                num_processes_resample = gr.Slider(
                    label="Número de Processos",
                    value=cpu_count() // 2,
                    minimum=1,
                    maximum=cpu_count(),
                    step=1,
                )
                normalize_resample = gr.Checkbox(
                    label="Normalizar volume do áudio",
                    value=False,
                )
                trim_resample = gr.Checkbox(
                    label="Remover silêncio no início e no fim do áudio",
                    value=False,
                )
                resample_btn = gr.Button(value="Executar", variant="primary")
                info_resample = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("### Passo 3: Pré-processamento de Transcrição")
                val_per_spk_manual = gr.Slider(
                    label="Dados de Validação por Falante",
                    value=0,
                    minimum=0,
                    maximum=100,
                    step=1,
                )
                max_val_total_manual = gr.Slider(
                    label="Máximo Total de Dados de Validação",
                    value=8,
                    minimum=0,
                    maximum=100,
                    step=1,
                )
                clean_manual = gr.Checkbox(
                    label="Ignorar arquivos ilegíveis (se desmarcado, interrompe com erro)",
                    value=False,
                )
                preprocess_text_btn = gr.Button(value="Executar", variant="primary")
                info_preprocess_text = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("### Passo 4: Geração de Características BERT")
                num_processes_bert = gr.Slider(
                    label="Número de Processos",
                    value=cpu_count() // 2,
                    minimum=1,
                    maximum=cpu_count(),
                    step=1,
                )
                bert_gen_btn = gr.Button(value="Executar", variant="primary")
                info_bert = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("### Passo 5: Geração de Características de Estilo")
                num_processes_style = gr.Slider(
                    label="Número de Processos",
                    value=cpu_count() // 2,
                    minimum=1,
                    maximum=cpu_count(),
                    step=1,
                )
                style_gen_btn = gr.Button(value="Executar", variant="primary")
                info_style = gr.Textbox(label="Status")

        gr.Markdown("## Treinamento")
        with gr.Row():
            skip_style = gr.Checkbox(
                label="Pular geração de arquivos de estilo",
                info="Marque esta opção se estiver retomando o treinamento",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="Usar versão JP-Extra",
                value=True,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="Desativar amostrador de lote personalizado",
                info="Marque se tiver VRAM de sobra; arquivos de áudio longos serão usados no treinamento",
                value=False,
            )
            speedup = gr.Checkbox(
                label="Acelerar treinamento pulando logs, etc.",
                value=False,
                visible=False,  # Experimental
            )
            train_btn = gr.Button(value="Iniciar Treinamento", variant="primary")
            tensorboard_btn = gr.Button(value="Abrir Tensorboard")
        gr.Markdown(
            "O progresso pode ser verificado no terminal. Os resultados são salvos a cada passo especificado, e o treinamento pode ser retomado do meio. Para encerrar o treinamento, basta fechar o terminal."
        )
        info_train = gr.Textbox(label="Status")

        preprocess_button.click(
            preprocess_all,
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                val_per_spk,
                max_val_total,
                log_interval,
                clean,
                bf16_run,
                fp16_run,
            ],
            outputs=[info_all],
        )

        # Manual preprocess
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                freeze_EN_bert_manual,
                freeze_JP_bert_manual,
                freeze_ZH_bert_manual,
                freeze_style_manual,
                freeze_decoder_manual,
                use_jp_extra_manual,
                log_interval_manual,
                bf16_run_manual,
                fp16_run_manual,
            ],
            outputs=[info_init],
        )
        resample_btn.click(
            second_elem_of(resample),
            inputs=[
                model_name,
                normalize_resample,
                trim_resample,
                num_processes_resample,
            ],
            outputs=[info_resample],
        )
        preprocess_text_btn.click(
            second_elem_of(preprocess_text),
            inputs=[
                model_name,
                use_jp_extra_manual,
                val_per_spk_manual,
                max_val_total_manual,
                clean_manual,
            ],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name, num_processes_bert],
            outputs=[info_bert],
        )
        style_gen_btn.click(
            second_elem_of(style_gen),
            inputs=[model_name, num_processes_style],
            outputs=[info_style],
        )

        # Train
        train_btn.click(
            second_elem_of(train),
            inputs=[
                model_name,
                skip_style,
                use_jp_extra_train,
                speedup,
                not_use_custom_batch_sampler,
            ],
            outputs=[info_train],
        )
        tensorboard_btn.click(
            run_tensorboard, inputs=[model_name], outputs=[tensorboard_btn]
        )

        use_jp_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra],
            outputs=[use_jp_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)
