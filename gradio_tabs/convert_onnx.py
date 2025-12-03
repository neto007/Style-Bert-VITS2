from pathlib import Path

import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils.subprocess import run_script_with_log


def call_convert_onnx(
    model: str,
):
    if model == "":
        return "Erro: Por favor, insira o nome do modelo."
    logger.info("Start converting model to onnx...")
    cmd = [
        "convert_onnx.py",
        "--model",
        model,
    ]
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Erro: {message}"
    return "A conversão para ONNX foi concluída."


initial_md = """
Converte modelos no formato safetensors para o formato ONNX.
Este modelo ONNX pode ser usado em bibliotecas externas compatíveis. Por exemplo, ao converter para o formato AIVM ou AIVMX com o [AIVM Generator](https://aivm-generator.aivis-project.com/), ele pode ser usado no [AivisSpeech](https://aivis-project.com/).

**A conversão leva cerca de 5 minutos ou mais**. Consulte o log do terminal para ver o progresso.

Após a conversão, será gerado um arquivo com o mesmo nome do modelo selecionado e extensão `.onnx`.
"""


def create_onnx_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def get_model_files(model_name: str):
        return [str(f) for f in model_holder.model_files_dict[model_name]]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"Modelo não encontrado. Por favor, coloque o modelo em {model_holder.root_dir}."
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Erro: Modelo não encontrado. Por favor, coloque o modelo em {model_holder.root_dir}."
            )
        return app
    initial_id = 0
    initial_pth_files = get_model_files(model_names[initial_id])

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Dropdown(
                    label="Lista de Modelos",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path = gr.Dropdown(
                    label="Arquivo do Modelo",
                    choices=initial_pth_files,
                    value=initial_pth_files[0],
                )
            refresh_button = gr.Button("Atualizar", scale=1)
            convert_button = gr.Button("Converter para ONNX", variant="primary", scale=1)
        text_output = gr.Textbox(label="Informações")

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        def refresh_fn() -> tuple[gr.Dropdown, gr.Dropdown]:
            names, files, _ = model_holder.update_model_names_for_gradio()
            return names, files

        refresh_button.click(
            refresh_fn,
            outputs=[model_name, model_path],
        )
        convert_button.click(
            call_convert_onnx,
            inputs=[model_path],
            outputs=[text_output],
        )

    return app


if __name__ == "__main__":
    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    model_holder = TTSModelHolder(assets_root, "cpu", "", ignore_onnx=True)
    app = create_onnx_app(model_holder)
    app.launch(inbrowser=True)
