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
    use_pt_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # Salvar log de pré-processamento em arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}, use_pt_extra: {use_pt_extra}"
    )

    default_config_path = "configs/config.json"
    if use_jp_extra:
        default_config_path = "configs/config_jp_extra.json"
    elif use_pt_extra:
        default_config_path = "configs/config_pt_extra.json"

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
    config["train"]["freeze_PT_bert"] = freeze_PT_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False  # デフォルトでFalseのはずだが念のため

    # 今はデフォルトであるが、以前は非JP-Extra版になくバグの原因になるので念のため
    config["data"]["use_jp_extra"] = use_jp_extra
    config["data"]["use_pt_extra"] = use_pt_extra

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
    pretrained_dir = Path("pretrained")
    if use_jp_extra:
        pretrained_dir = Path("pretrained_jp_extra")
    elif use_pt_extra:
        pretrained_dir = Path("pretrained_pt_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Step 1: {pretrained_dir} folder not found.")
        return False, f"Step 1, Error: A pasta {pretrained_dir} não foi encontrada."

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
    logger.success("Step 1: initialization finished.")
    return True, "Step 1, Success: Configuração inicial concluída"


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Step 2: start resampling...")
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
        logger.error("Step 2: resampling failed.")
        return False, f"Step 2, Error: Falha no pré-processamento do arquivo de áudio:\n{message}"
    elif message:
        logger.warning("Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: Pré-processamento do arquivo de áudio concluído:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Step 2, Success: Pré-processamento do arquivo de áudio concluído"


def preprocess_text(
    model_name: str, use_jp_extra: bool, use_pt_extra: bool, val_per_lang: int, yomi_error: str
):
    logger.info("Step 3: start preprocessing text...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Step 3: {paths.esd_path} not found.")
        return (
            False,
            f"Step 3, Error: Arquivo de transcrição {paths.esd_path} não encontrado.",
        )

    cmd = [
        "preprocess_text.py",
        "--config-path",
        str(paths.config_path),
        "--transcription-path",
        str(paths.esd_path),
        "--train-path",
        str(paths.train_path),
        "--val-path",
        str(paths.val_path),
        "--val-per-lang",
        str(val_per_lang),
        "--yomi_error",
        yomi_error,
        "--correct_path",  # 音声ファイルのパスを正しいパスに修正する
    ]
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    if use_pt_extra:
        cmd.append("--use_pt_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 3: preprocessing text failed.")
        return (
            False,
            f"Step 3, Error: Falha no pré-processamento do arquivo de transcrição:\n{message}",
        )
    elif message:
        logger.warning("Step 3: preprocessing text finished with stderr.")
        return (
            True,
            f"Step 3, Success: Pré-processamento do arquivo de transcrição concluído:\n{message}",
        )
    logger.success("Step 3: preprocessing text finished.")
    return True, "Step 3, Success: Pré-processamento do arquivo de transcrição concluído"


def bert_gen(model_name: str):
    logger.info("Step 4: start bert_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path)]
    )
    if not success:
        logger.error("Step 4: bert_gen failed.")
        return False, f"Step 4, Error: Falha na geração do arquivo de características BERT:\n{message}"
    elif message:
        logger.warning("Step 4: bert_gen finished with stderr.")
        return (
            True,
            f"Step 4, Success: Geração do arquivo de características BERT concluída:\n{message}",
        )
    logger.success("Step 4: bert_gen finished.")
    return True, "Step 4, Success: Geração do arquivo de características BERT concluída"


def style_gen(model_name: str, num_processes: int):
    logger.info("Step 5: start style_gen...")
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
        logger.error("Step 5: style_gen failed.")
        return (
            False,
            f"Step 5, Error: Falha na geração do arquivo de características de estilo:\n{message}",
        )
    elif message:
        logger.warning("Step 5: style_gen finished with stderr.")
        return (
            True,
            f"Step 5, Success: Geração do arquivo de características de estilo concluída:\n{message}",
        )
    logger.success("Step 5: style_gen finished.")
    return True, "Step 5, Success: Geração do arquivo de características de estilo concluída"


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_PT_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    use_pt_extra: bool,
    val_per_lang: int,
    log_interval: int,
    yomi_error: str,
):
    if model_name == "":
        return False, "Error: Por favor, insira o nome do modelo"
    success, message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_PT_bert=freeze_PT_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        use_pt_extra=use_pt_extra,
        log_interval=log_interval,
    )
    if not success:
        return False, message
    success, message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if not success:
        return False, message

    success, message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        use_pt_extra=use_pt_extra,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
    )
    if not success:
        return False, message
    success, message = bert_gen(
        model_name=model_name
    )  # bert_genは重いのでプロセス数いじらない
    if not success:
        return False, message
    success, message = style_gen(model_name=model_name, num_processes=num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return (
        True,
        "Success: Todo o pré-processamento foi concluído. Recomendamos verificar o terminal para ver se há algo estranho.",
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    use_pt_extra: bool = False,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    # 学習再開の場合を考えて念のためconfig.ymlの名前等を更新
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
        logger.error("Train failed.")
        return False, f"Error: Falha no treinamento:\n{message}"
    elif message:
        logger.warning("Train finished with stderr.")
        return True, f"Success: Treinamento concluído:\n{message}"
    logger.success("Train finished.")
    return True, "Success: Treinamento concluído"


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True  # ポートが開いている場合
        except OSError:
            pass  # ポートがまだ開いていない場合

        if time.time() - start_time > timeout:
            return False  # タイムアウト

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
        yield gr.Button("Iniciando...")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard did not start in the expected time.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Abrir Tensorboard")


change_log_md = """
**Mudanças na Ver 2.5 e posteriores**

- Ao colocar o áudio em subdiretórios dentro da pasta `raw/`, os estilos agora são criados automaticamente. Veja "Como usar/Preparação de dados" abaixo para detalhes.
- Anteriormente, arquivos de áudio com mais de 14 segundos não eram usados para treinamento, mas na Ver 2.5 e posteriores, você pode treinar sem essa limitação marcando "Desativar Amostrador de Batch Personalizado" (padrão é desligado). No entanto:
    - A eficiência do treinamento pode ser ruim se os arquivos de áudio forem longos, e o comportamento não foi verificado.
    - Marcar essa opção aumentará significativamente a VRAM necessária, então se o treinamento falhar ou faltar VRAM, diminua o tamanho do batch ou desmarque a opção.
"""

how_to_md = """
## Como usar

- Prepare os dados, insira o nome do modelo, ajuste as configurações se necessário e clique no botão "Executar Pré-processamento Automático". O progresso será exibido no terminal.

- Use "Pré-processamento Manual" se quiser executar cada passo individualmente (basicamente o automático deve servir).

- Quando o pré-processamento terminar, clique no botão "Iniciar Treinamento" para começar o treinamento.

- Para retomar o treinamento do meio, basta inserir o nome do modelo e clicar em "Iniciar Treinamento".

## Sobre a versão JP-Extra

Você pode usar [Bert-VITS2 Japanese-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta) como estrutura de modelo base.
Tende a melhorar o sotaque, entonação e naturalidade do japonês, mas perde a capacidade de falar inglês e chinês.
"""

prepare_md = """
Primeiro, prepare os dados de áudio e o texto de transcrição.

Organize-os da seguinte maneira:
```
├── Data/
│   ├── {nome_do_modelo}
│   │   ├── esd.list
│   │   ├── raw/
│   │   │   ├── foo.wav
│   │   │   ├── bar.mp3
│   │   │   ├── style1/
│   │   │   │   ├── baz.wav
│   │   │   │   ├── qux.wav
│   │   │   ├── style2/
│   │   │   │   ├── corge.wav
│   │   │   │   ├── grault.wav
...
```

### Como organizar
- Se organizado como acima, além do estilo padrão, os estilos `style1` e `style2` serão criados automaticamente a partir dos arquivos de áudio dentro das pastas `style1/` e `style2/` (incluindo subpastas).
- Se não precisar criar estilos específicos, ou se for criar estilos usando a função de classificação de estilo, coloque tudo diretamente na pasta `raw/`. Se o número de subdiretórios em `raw/` for 0 ou 1, apenas o estilo padrão será criado.
- O formato do arquivo de áudio suporta muitos formatos como mp3 além do formato wav.

### Arquivo de transcrição `esd.list`

No arquivo `Data/{nome_do_modelo}/esd.list`, descreva as informações de cada arquivo de áudio no seguinte formato:


```
path/to/audio.wav(mesmo se não for wav)|{nome_do_falante}|{ID_do_idioma, ZH, JP ou EN}|{texto_de_transcrição}
```

- Aqui, o primeiro `path/to/audio.wav` é o caminho relativo a partir de `raw/`. Ou seja, para `raw/foo.wav` é `foo.wav`, e para `raw/style1/bar.wav` é `style1/bar.wav`.
- Mesmo que a extensão não seja wav, escreva `wav` no `esd.list`. Por exemplo, para `raw/bar.mp3`, escreva `bar.wav`.


Exemplo:
```
foo.wav|hanako|JP|Olá, como vai?
bar.wav|taro|JP|Sim, estou ouvindo... Precisa de algo?
style1/baz.wav|hanako|JP|O tempo está bom hoje, não é?
style1/qux.wav|taro|JP|Sim, é verdade.
...
english_teacher.wav|Mary|EN|How are you? I'm fine, thank you, and you?
...
```
Claro, um conjunto de dados de um único falante japonês também é aceitável.
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("Como usar", open=False):
            gr.Markdown(how_to_md)
            with gr.Accordion(label="Preparação de dados", open=False):
                gr.Markdown(prepare_md)
        model_name = gr.Textbox(label="Nome do Modelo")
        gr.Markdown("### Pré-processamento Automático")
        with gr.Row(variant="panel"):
            with gr.Column():
                use_jp_extra = gr.Checkbox(
                    label="Usar versão JP-Extra (melhora desempenho em japonês, mas perde capacidade de falar inglês e chinês)",
                    value=False,
                )
                use_pt_extra = gr.Checkbox(
                    label="Usar versão PT-Extra (melhora desempenho em português)",
                    value=True,
                )
                batch_size = gr.Slider(
                    label="Tamanho do Batch",
                    info="Se a velocidade de treinamento for lenta, tente diminuir. Se tiver VRAM de sobra, aumente. Estimativa de uso de VRAM na versão JP-Extra: 1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB",
                    value=2,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs = gr.Slider(
                    label="Número de Épocas",
                    info="100 parece suficiente, mas mais épocas podem melhorar a qualidade",
                    value=100,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                save_every_steps = gr.Slider(
                    label="Salvar resultados a cada X passos",
                    info="Note que é diferente do número de épocas",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                normalize = gr.Checkbox(
                    label="Normalizar volume do áudio (se o volume não estiver consistente)",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="Remover silêncio no início e fim do áudio",
                    value=False,
                )
                yomi_error = gr.Radio(
                    label="Como lidar com arquivos de transcrição ilegíveis",
                    choices=[
                        ("Interromper ao final do pré-processamento de texto se ocorrer erro", "raise"),
                        ("Continuar sem usar arquivos ilegíveis", "skip"),
                        ("Forçar leitura de arquivos ilegíveis e usar no treinamento", "use"),
                    ],
                    value="skip",
                )
                with gr.Accordion("Configurações Detalhadas", open=False):
                    num_processes = gr.Slider(
                        label="Número de Processos",
                        info="Número de processos paralelos no pré-processamento. Diminua se travar.",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    val_per_lang = gr.Slider(
                        label="Número de Dados de Validação",
                        info="Não usado para treinamento, mas para comparar áudio original e sintetizado no tensorboard",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    log_interval = gr.Slider(
                        label="Intervalo de Log do Tensorboard",
                        info="Diminua se quiser ver detalhes no Tensorboard",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    gr.Markdown("Congelar partes específicas durante o treinamento?")
                    freeze_EN_bert = gr.Checkbox(
                        label="Congelar parte BERT Inglês",
                        value=False,
                    )
                    freeze_PT_bert = gr.Checkbox(
                        label="Congelar parte BERT Português",
                        value=False,
                    )
                    freeze_JP_bert = gr.Checkbox(
                        label="Congelar parte BERT Japonês",
                        value=False,
                    )
                    freeze_ZH_bert = gr.Checkbox(
                        label="Congelar parte BERT Chinês",
                        value=False,
                    )
                    freeze_style = gr.Checkbox(
                        label="Congelar parte de Estilo",
                        value=False,
                    )
                    freeze_decoder = gr.Checkbox(
                        label="Congelar parte do Decodificador",
                        value=False,
                    )

            with gr.Column():
                preprocess_button = gr.Button(
                    value="Executar Pré-processamento Automático", variant="primary"
                )
                info_all = gr.Textbox(label="Status")
        with gr.Accordion(open=False, label="Pré-processamento Manual"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Passo 1: Gerar Arquivo de Configuração")
                    use_jp_extra_manual = gr.Checkbox(
                        label="Usar versão JP-Extra",
                        value=False,
                    )
                    use_pt_extra_manual = gr.Checkbox(
                        label="Usar versão PT-Extra",
                        value=True,
                    )
                    batch_size_manual = gr.Slider(
                        label="Tamanho do Batch",
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
                        label="Salvar resultados a cada X passos",
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
                    freeze_EN_bert_manual = gr.Checkbox(
                        label="Congelar parte BERT Inglês",
                        value=False,
                    )
                    freeze_JP_bert_manual = gr.Checkbox(
                        label="Congelar parte BERT Japonês",
                        value=False,
                    )
                    freeze_ZH_bert_manual = gr.Checkbox(
                        label="Congelar parte BERT Chinês",
                        value=False,
                    )
                    freeze_style_manual = gr.Checkbox(
                        label="Congelar parte de Estilo",
                        value=False,
                    )
                    freeze_decoder_manual = gr.Checkbox(
                        label="Congelar parte do Decodificador",
                        value=False,
                    )
                with gr.Column():
                    generate_config_btn = gr.Button(value="Executar", variant="primary")
                    info_init = gr.Textbox(label="Status")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Passo 2: Pré-processamento de Arquivos de Áudio")
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
                        label="Remover silêncio no início e fim do áudio",
                        value=False,
                    )
                with gr.Column():
                    resample_btn = gr.Button(value="Executar", variant="primary")
                    info_resample = gr.Textbox(label="Status")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Passo 3: Pré-processamento de Arquivos de Transcrição")
                    val_per_lang_manual = gr.Slider(
                        label="Número de Dados de Validação",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    yomi_error_manual = gr.Radio(
                        label="Como lidar com arquivos de transcrição ilegíveis",
                        choices=[
                            ("Interromper ao final do pré-processamento de texto se ocorrer erro", "raise"),
                            ("Continuar sem usar arquivos ilegíveis", "skip"),
                            ("Forçar leitura de arquivos ilegíveis e usar no treinamento", "use"),
                        ],
                        value="raise",
                    )
                with gr.Column():
                    preprocess_text_btn = gr.Button(value="Executar", variant="primary")
                    info_preprocess_text = gr.Textbox(label="Status")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Passo 4: Gerar Arquivo de Características BERT")
                with gr.Column():
                    bert_gen_btn = gr.Button(value="Executar", variant="primary")
                    info_bert = gr.Textbox(label="Status")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Passo 5: Gerar Arquivo de Características de Estilo")
                    num_processes_style = gr.Slider(
                        label="Número de Processos",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                with gr.Column():
                    style_gen_btn = gr.Button(value="Executar", variant="primary")
                    info_style = gr.Textbox(label="Status")
        gr.Markdown("## Treinamento")
        with gr.Row():
            skip_style = gr.Checkbox(
                label="Pular geração de arquivo de estilo",
                info="Marque se estiver retomando o treinamento",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="Usar versão JP-Extra",
                value=False,
            )
            use_pt_extra_train = gr.Checkbox(
                label="Usar versão PT-Extra",
                value=True,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="Desativar Amostrador de Batch Personalizado",
                info="Marque se tiver VRAM sobrando para usar arquivos de áudio longos no treinamento",
                value=False,
            )
            speedup = gr.Checkbox(
                label="Pular logs etc. para acelerar o treinamento",
                value=False,
                visible=False,  # Experimental
            )
            train_btn = gr.Button(value="Iniciar Treinamento", variant="primary")
            tensorboard_btn = gr.Button(value="Abrir Tensorboard")
        gr.Markdown(
            "Verifique o progresso no terminal. Os resultados são salvos a cada passo especificado, e você pode retomar o treinamento do meio. Para encerrar o treinamento, basta fechar o terminal."
        )
        info_train = gr.Textbox(label="Status")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_PT_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                use_pt_extra,
                val_per_lang,
                log_interval,
                yomi_error,
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
                use_pt_extra_manual,
                log_interval_manual,
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
                use_pt_extra_manual,
                val_per_lang_manual,
                yomi_error_manual,
            ],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name],
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
                use_pt_extra_train,
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
        use_pt_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_pt_extra],
            outputs=[use_pt_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )
        use_pt_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_pt_extra_manual],
            outputs=[use_pt_extra_train],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)
