import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Erro: Por favor, insira o nome do modelo."
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # Ignorar avisos do onnx
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "O fatiamento do áudio foi concluído."


def do_transcribe(
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Erro: Por favor, insira o nome do modelo."
    if hf_repo_id == "litagin/anime-whisper":
        logger.info(
            "Since litagin/anime-whisper does not support initial prompt, it will be ignored."
        )
        initial_prompt = ""

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Erro: {message}. Se a mensagem de erro estiver vazia, pode não haver problemas, então verifique o arquivo de transcrição e ignore se estiver tudo bem."
    return "A transcrição do áudio foi concluída."


how_to_md = """
Esta é uma ferramenta para criar conjuntos de dados de treinamento para o Style-Bert-VITS2. Consiste nas duas partes a seguir:

- Cortar e fatiar seções de fala de comprimento apropriado do áudio fornecido
- Transcrição do áudio

Você pode usar ambos, ou apenas o último se não precisar fatiar. **Se você já tem arquivos de áudio de comprimento apropriado, como fontes de corpus, o fatiamento não é necessário**.

## O que é necessário

Alguns arquivos de áudio contendo a voz que você deseja treinar (formatos como mp3 além de wav são possíveis).
É melhor ter uma certa quantidade de tempo total, há relatos de que 10 minutos são suficientes. Pode ser um único arquivo ou vários arquivos.

## Como usar o Fatiamento
1. Coloque todos os arquivos de áudio na pasta `inputs` (se quiser separar por estilo, coloque os áudios em subpastas para cada estilo)
2. Insira o `Nome do modelo`, ajuste as configurações se necessário e clique no botão `Executar Fatiamento`
3. Os arquivos de áudio resultantes serão salvos em `Data/{Nome do modelo}/raw`

## Como usar a Transcrição

1. Verifique se os arquivos de áudio estão em `Data/{Nome do modelo}/raw` (não precisam estar diretamente na raiz)
2. Ajuste as configurações se necessário e clique no botão
3. O arquivo de transcrição será salvo em `Data/{Nome do modelo}/esd.list`

## Notas

- ~~Arquivos wav muito longos (mais de 12-15 segundos?) parecem não ser usados para treinamento. Também pode não ser bom se forem muito curtos.~~ Esta limitação desapareceu na Ver 2.5 se você selecionar "Não usar amostrador de lote personalizado" durante o treinamento. No entanto, áudios muito longos podem aumentar o consumo de VRAM ou tornar o treinamento instável, por isso recomendamos fatiar para um comprimento apropriado.
- O quanto você precisa corrigir o resultado da transcrição depende do conjunto de dados.
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "**Se você já tem uma coleção de arquivos de áudio de cerca de 2-12 segundos por arquivo e seus dados de transcrição, pode treinar sem usar esta guia.**"
        )
        with gr.Accordion("Como usar", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="Por favor, insira o nome do modelo (também será usado como nome do falante)."
        )
        with gr.Accordion("Fatiamento de Áudio"):
            gr.Markdown(
                "**Se você já tem dados consistindo em arquivos de áudio de comprimento apropriado, coloque esses áudios em Data/{Nome do modelo}/raw e esta etapa não é necessária.**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="Caminho da pasta contendo o áudio original",
                        value="inputs",
                        info="Por favor, coloque arquivos wav, mp3, etc. na pasta abaixo",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="Cortar se for menor que estes segundos",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="Cortar se for maior que estes segundos",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="Duração mínima do silêncio para considerar como silêncio (ms)",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="Adicionar o intervalo de tempo do arquivo original ao final do nome do arquivo WAV",
                    )
                    slice_button = gr.Button("Executar Fatiamento")
                result1 = gr.Textbox(label="Resultado")
        with gr.Row():
            with gr.Column():
                use_hf_whisper = gr.Checkbox(
                    label="Usar Whisper do HuggingFace (mais rápido, mas usa mais VRAM)",
                    value=False,
                )
                whisper_model = gr.Dropdown(
                    [
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Modelo Whisper",
                    value="large-v3",
                    visible=True,
                )
                hf_repo_id = gr.Dropdown(
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-large-v3",
                        "openai/whisper-large-v2",
                        "kotoba-tech/kotoba-whisper-v2.1",
                        "litagin/anime-whisper",
                    ],
                    label="Whisper repo_id do HuggingFace",
                    value="openai/whisper-large-v3-turbo",
                    visible=False,
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="Precisão de cálculo",
                    value="bfloat16",
                    visible=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="Tamanho do lote",
                    info="Aumentar torna mais rápido, mas usa mais VRAM",
                    visible=False,
                )
                language = gr.Dropdown(["ja", "en", "zh","pt-br"], value="ja", label="Idioma")
                initial_prompt = gr.Textbox(
                    label="Prompt inicial",
                    value="Olá. Como você está? Hehe, eu estou... muito bem!",
                    info="Exemplo de frase que você deseja que seja transcrita dessa maneira (pontuação, risadas, nomes próprios, etc.)",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Número de feixes do Beam Search",
                    info="Quanto menor, mais rápido (anteriormente era 5)",
                )
            transcribe_button = gr.Button("Transcrição de Áudio")
            result2 = gr.Textbox(label="Resultado")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )
        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=not x),
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[whisper_model, hf_repo_id, batch_size, compute_type],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)
