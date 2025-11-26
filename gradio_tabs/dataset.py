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
        return "Error: モデル名を入力してください。"
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
    # Ignora avisos do ONNX
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "Divisão de áudio concluída."


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
        return "Error: Insira o nome do modelo."
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
        return f"Erro: {message}. Se a mensagem de erro estiver vazia, pode não haver problema; verifique o arquivo de transcrição e ignore se tudo estiver bem."
    return "Transcrição de áudio concluída."


how_to_md = """
Ferramenta para criar conjuntos de dados de treinamento para Style-Bert-VITS2. Consiste em duas partes.

- Recorta intervalos de fala de comprimento adequado a partir do áudio fornecido.
- Transcrição do áudio.

Você pode usar ambos ou apenas a transcrição se não precisar dividir. **Se já houver arquivos de áudio de comprimento adequado, a divisão não é necessária.**

## Requisitos

Alguns arquivos de áudio contendo a voz que você deseja treinar (formatos como wav, mp3 etc. são aceitos).
É recomendável ter um tempo total razoável, relatos indicam que 10 minutos já são suficientes. Pode ser um único arquivo ou vários.

## Como usar a divisão
1. Coloque todos os arquivos de áudio na pasta `inputs` (se quiser separar por estilo, coloque-os em subpastas por estilo).
2. Insira o `nome do modelo`, ajuste as configurações se necessário e clique no botão `Dividir áudio`.
3. Os arquivos de áudio resultantes serão salvos em `Data/{nome_do_modelo}/raw`.

## Como usar a transcrição

1. Verifique que os arquivos de áudio estão em `Data/{nome_do_modelo}/raw` (não precisa ser diretamente na raiz).
2. Ajuste as configurações se necessário e clique no botão.
3. O arquivo de transcrição será salvo em `Data/{nome_do_modelo}/esd.list`.

## Avisos

- ~~Arquivos wav muito longos (mais que 12‑15 s?) não são usados no treinamento; arquivos muito curtos também podem ser problemáticos.~~ Essa restrição foi removida na versão 2.5 ao desativar o “custom batch sampler”. Contudo, áudio muito longo pode consumir muita VRAM ou causar instabilidade, então recomenda‑se dividir em comprimentos adequados.
- O quanto será necessário corrigir a transcrição depende do conjunto de dados.
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "**Se já houver um conjunto de arquivos de áudio de 2‑12 s e suas transcrições, você pode treinar sem usar esta aba.**"
        )
        with gr.Accordion("Como usar", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="Insira o nome do modelo (também usado como nome do locutor)."
        )
        with gr.Accordion("Divisão de áudio"):
            gr.Markdown(
                "**Se já houver dados com arquivos de áudio de comprimento adequado, basta colocá‑los em Data/{nome_do_modelo}/raw e esta etapa não será necessária.**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="Caminho da pasta contendo o áudio original",
                        value="inputs",
                        info="Coloque arquivos wav, mp3 etc. na pasta abaixo",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="Descartar trechos menores que este número de segundos",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="Descartar trechos maiores que este número de segundos",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="Duração mínima (ms) de silêncio para considerar como separação",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="Anexar intervalo de tempo original ao final do nome do arquivo WAV",
                    )
                    slice_button = gr.Button("Executar divisão")
                result1 = gr.Textbox(label="Resultado")
        with gr.Row():
            with gr.Column():
                use_hf_whisper = gr.Checkbox(
                    label="Usar Whisper da HuggingFace (rápido, mas consome mais VRAM)",
                    value=False,
                )
                whisper_model = gr.Dropdown(
                    [
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Whisper modelo",
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
                    label="HuggingFace Whisper repo_id",
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
                    info="Aumentar melhora a velocidade, mas usa mais VRAM",
                    visible=False,
                )
                language = gr.Dropdown(["pt", "en", "zh"], value="pt", label="Idioma")
                initial_prompt = gr.Textbox(
                    label="Prompt inicial",
                    value="Olá! Como você está? Haha, eu estou bem!",
                    info="Exemplo de como você gostaria que a transcrição fosse feita (pontuação, risadas, nomes próprios, etc.)",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Número de feixes no Beam Search",
                    info="Valores menores aumentam a velocidade (antes era 5)",
                )
            transcribe_button = gr.Button("Transcrever áudio")
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
