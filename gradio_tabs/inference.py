import datetime
import json
from pathlib import Path
from typing import Optional

import gradio as gr

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers


# Inicializar pyopenjtalk_worker
## pyopenjtalk_worker é um servidor de socket TCP, então iniciamos aqui
pyopenjtalk.initialize_worker()

# Para evitar consumo desnecessário de GPU VRAM durante o treinamento na Web UI, não pré-carregamos o modelo BERT aqui
# As características BERT do dataset são extraídas previamente por bert_gen.py, então não é necessário carregar o modelo BERT durante o treinamento
# O pré-carregamento do modelo BERT é feito dentro do TTSModelHolder.get_model_for_gradio() quando o botão 'Carregar' é pressionado
# Durante o treinamento na Web UI, se não pressionar o botão 'Carregar' na aba de síntese de voz, pode iniciar o treinamento sem o modelo BERT carregado na VRAM

languages = [lang.value for lang in Languages]

initial_text = "Olá, prazer em conhecê-lo. Qual é o seu nome?"

examples = [
    [initial_text, "JP"],
    [
        """Você dizer isso me deixa muito feliz.
Você dizer isso me deixa muito irritada.
Você dizer isso me deixa muito surpresa.
Você dizer isso me deixa muito triste.""",
        "JP",
    ],
    [  # Frase de confissão criada com ChatGPT
        """Eu tenho observado você há muito tempo. Fui cativada pelo seu sorriso, sua gentileza, sua força.
Passando tempo juntos como amigos, percebi que você foi se tornando alguém especial para mim.
Então... eu gosto de você! Se não se importar, você quer namorar comigo?""",
        "JP",
    ],
    [  # Natsume Soseki - Eu sou um gato
        """Eu sou um gato. Ainda não tenho nome.
Não faço ideia de onde nasci. Só me lembro de estar miando em um lugar escuro e úmido.
Foi aqui que vi um ser humano pela primeira vez. Mais tarde descobri que era um estudante, aparentemente a espécie mais feroz entre os humanos.
Dizem que esses estudantes às vezes nos capturam e nos cozinham.""",
        "JP",
    ],
    [  # Kajii Motojiro - Sob as cerejeiras
        """Há cadáveres enterrados sob as cerejeiras! Você pode acreditar nisso.
Por que outra razão as flores de cerejeira floresceriam tão magnificamente? Não conseguia acreditar naquela beleza, então fiquei ansioso nos últimos dias.
Mas agora, finalmente entendi. Há cadáveres enterrados sob as cerejeiras. Você pode acreditar nisso.""",
        "JP",
    ],
    [  # Frases emocionais criadas com ChatGPT
        """Eba! Tirei nota máxima no teste! Estou muito feliz!
Por que você ignora minha opinião? Não posso perdoar! Que raiva! Você deveria sumir!
Hahahaha! Esse mangá é muito engraçado, olha isso, hehehe, hahaha.
Você se foi e fiquei sozinha, tão triste que quase choro.""",
        "JP",
    ],
    [  # Versão formal da anterior
        """Consegui! Tirei nota máxima no teste! Estou muito feliz!
Por que você ignora minha opinião? Não posso perdoar! Que raiva! Desapareça!
Hahahaha! Esse mangá é muito engraçado, veja isso, hehehe, hahaha.
Você se foi e fiquei sozinha, tão triste que quase choro.""",
        "JP",
    ],
    [  # Explicação sobre síntese de voz criada com ChatGPT
        """A síntese de voz é uma tecnologia que usa aprendizado de máquina para reproduzir a voz humana a partir de texto. Esta tecnologia analisa a estrutura da linguagem e gera áudio com base nisso.
Com os últimos avanços nesta área, é possível gerar vozes mais naturais e expressivas. A aplicação de aprendizado profundo permite reproduzir até mudanças sutis na qualidade vocal, incluindo emoções e sotaques.""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "A síntese de voz é a produção artificial da fala humana. Sistemas de computador usados para este propósito são chamados de sintetizadores de voz, e podem ser implementados via software ou hardware.",
        "ZH",
    ],
]

initial_md = """
- O modelo padrão [`koharune-ami` (Koharune Ami)](https://huggingface.co/litagin/sbv2_koharune_ami) e o modelo [`amitaro` (Amitaro)](https://huggingface.co/litagin/sbv2_amitaro) adicionados na Ver 2.5 são modelos treinados com permissão prévia usando fontes de corpus e áudio de transmissão ao vivo publicados no [Amitaro's Voice Material Workshop](https://amitaro.net/). Por favor, certifique-se de **ler os termos de uso** abaixo antes de usar.

- Para baixar os modelos acima após a atualização da Ver 2.5, clique duas vezes em `Initialize.bat` ou baixe manualmente e coloque-os no diretório `model_assets`.

- A **versão do editor** adicionada na Ver 2.3 pode ser mais fácil de usar para leitura real. Você pode iniciá-la com `Editor.bat` ou `python server_editor.py --inbrowser`.
"""

terms_of_use_md = """
## Solicitações e Licença do Modelo Padrão

Consulte [aqui](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md) para as solicitações e termos de uso mais recentes. A versão mais recente é sempre aplicada.

Ao usar o Style-Bert-VITS2, agradeceríamos se você pudesse seguir as solicitações abaixo. No entanto, as partes anteriores aos termos de uso do modelo são apenas "solicitações" e não têm força obrigatória, e não são os termos de uso do Style-Bert-VITS2. Portanto, não contradiz a [licença do repositório](https://github.com/litagin02/Style-Bert-VITS2#license), e apenas a licença do repositório tem força obrigatória ao usar o repositório.

### O que não queremos que você faça

Não queremos que você use o Style-Bert-VITS2 para os seguintes fins:

- Fins que violem a lei
- Fins políticos (proibido no Bert-VITS2 original)
- Fins que prejudiquem os outros
- Fins de personificação ou criação de deepfakes

### O que queremos que você cumpra

- Ao usar o Style-Bert-VITS2, certifique-se de verificar os termos de uso e a licença do modelo que está usando e siga-os, se existirem.
- Além disso, ao usar o código-fonte, siga a [licença do repositório](https://github.com/litagin02/Style-Bert-VITS2#license).

Abaixo está a licença para os modelos incluídos por padrão.

### Corpus JVNV (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- A licença do [Corpus JVNV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) é [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja), portanto, herdamos isso.

### Koharune Ami (koharune-ami) / Amitaro (amitaro)

Você deve cumprir todos os [termos do Amitaro's Voice Material Workshop](https://amitaro.net/voice/voice_rule/) e [termos de uso de áudio de transmissão ao vivo do Amitaro](https://amitaro.net/voice/livevoice/#index_id6). Em particular, observe os seguintes itens (você pode usar para fins comerciais ou não comerciais se seguir os termos):

#### Proibições

- Uso em trabalhos/aplicações com restrição de idade
- Trabalhos/aplicações profundamente relacionados a novas religiões, política, marketing multinível, etc.
- Trabalhos/aplicações que caluniem organizações, indivíduos ou nações específicas
- Tratar o áudio gerado como a voz do próprio Amitaro
- Tratar o áudio gerado como a voz de alguém que não seja Amitaro

#### Créditos

Ao publicar o áudio gerado (independentemente da mídia), certifique-se de incluir uma notação de crédito em um local fácil de entender, indicando que você está usando um modelo de voz baseado na voz de `Amitaro's Voice Material Workshop (https://amitaro.net/)`.

Exemplo de crédito:
- `Modelo Style-BertVITS2: Koharune Ami, Amitaro's Voice Material Workshop (https://amitaro.net/)`
- `Modelo Style-BertVITS2: Amitaro, Amitaro's Voice Material Workshop (https://amitaro.net/)`

#### Fusão de Modelos

Para fusão de modelos, cumpra as [respostas às perguntas frequentes do Amitaro's Voice Material Workshop](https://amitaro.net/voice/faq/#index_id17):
- Este modelo só pode ser mesclado com outro modelo se o detentor dos direitos da voz usada para treinar esse outro modelo permitir.
- Se as características da voz de Amitaro permanecerem (se a taxa de fusão for de 25% ou mais), o uso será limitado ao escopo dos [termos do Amitaro's Voice Material Workshop](https://amitaro.net/voice/voice_rule/), e esses termos também se aplicarão a esse modelo.
"""

how_to_md = """
Coloque os arquivos do modelo dentro do diretório `model_assets` como mostrado abaixo.
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
Cada modelo precisa dos seguintes arquivos:
- `config.json`: Arquivo de configuração de treinamento
- `*.safetensors`: Arquivo do modelo treinado (pelo menos um é necessário, vários são permitidos)
- `style_vectors.npy`: Arquivo de vetores de estilo

Os dois primeiros são salvos automaticamente no local correto durante o treinamento com `Train.bat`. `style_vectors.npy` deve ser gerado executando `Style.bat` e seguindo as instruções.
"""

style_md = f"""
- Você pode controlar o tom, emoção e estilo da leitura a partir de predefinições ou arquivos de áudio.
- Mesmo com o padrão {DEFAULT_STYLE}, a leitura será expressiva com emoções adequadas à frase. Este controle de estilo é como sobrescrever isso com um peso.
- Se a intensidade for muito alta, a pronúncia pode ficar estranha ou a voz pode falhar.
- A intensidade ideal parece variar dependendo do modelo e do estilo.
- Ao inserir um arquivo de áudio, pode não ter um bom efeito a menos que seja um falante com um tom semelhante aos dados de treinamento (especialmente do mesmo gênero).
"""
voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]


def make_interactive():
    return gr.update(interactive=True, value="Síntese de Voz")


def make_non_interactive():
    return gr.update(interactive=False, value="Síntese de Voz (Por favor, carregue o modelo)")


def gr_util(item):
    if item == "Escolher da predefinição":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


null_models_frame = 0


def change_null_model_row(
    null_model_index: int,
    null_model_name: str,
    null_model_path: str,
    null_voice_weights: float,
    null_voice_pitch_weights: float,
    null_speech_style_weights: float,
    null_tempo_weights: float,
    null_models: dict[int, NullModelParam],
):
    null_models[null_model_index] = NullModelParam(
        name=null_model_name,
        path=Path(null_model_path),
        weight=null_voice_weights,
        pitch=null_voice_pitch_weights,
        style=null_speech_style_weights,
        tempo=null_tempo_weights,
    )
    if len(null_models) > null_models_frame:
        keys_to_keep = list(range(null_models_frame))
        result = {k: null_models[k] for k in keys_to_keep}
    else:
        result = null_models
    return result, True


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
        text,
        language,
        reference_audio_path,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        line_split,
        split_interval,
        assist_text,
        assist_text_weight,
        use_assist_text,
        style,
        style_weight,
        kata_tone_json_str,
        use_tone,
        speaker,
        pitch_scale,
        intonation_scale,
        null_models: dict[int, NullModelParam],
        force_reload_model: bool,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None
        logger.debug(f"Null models setting: {null_models}")

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "A especificação de acento é suportada apenas em japonês."
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "A especificação de acento é suportada apenas quando não se usa geração dividida por quebras de linha."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # Converter para usar tuple
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"A especificação de acento é inválida: {e}"
                kata_tone = None

        # tone só se torna not None quando é realmente passado para a síntese de voz
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = model_holder.current_model.spk2id[speaker]

        start_time = datetime.datetime.now()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise_scale,
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
                null_model_params=null_models,
                force_reload_model=force_reload_model,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Erro: A especificação de acento é inválida:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Erro: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # Retorna informações de acento para uso na especificação de acentos
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Success, time: {duration} seconds."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str, False

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
        gr.Markdown(terms_of_use_md)
        null_models = gr.State({})
        force_reload_model = gr.State(False)
        with gr.Accordion(label="Como usar", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
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
                    refresh_button = gr.Button("Atualizar", scale=1, visible=True)
                    load_button = gr.Button("Carregar", scale=1, variant="primary")
                text_input = gr.TextArea(label="Texto", value=initial_text)
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="Tom (valores diferentes de 1 degradam a qualidade)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="Entonação (valores diferentes de 1 degradam a qualidade)",
                )

                line_split = gr.Checkbox(
                    label="Gerar dividido por quebras de linha (dividir transmite melhor a emoção)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="Duração do silêncio entre quebras de linha (segundos)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="Ajuste de acento (apenas 0=baixo ou 1=alto)",
                    info="Disponível apenas quando não dividido por quebras de linha. Não é perfeito.",
                )
                use_tone = gr.Checkbox(label="Usar ajuste de acento", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="JP", label="Idioma")
                speaker = gr.Dropdown(label="Falante")
                with gr.Accordion(label="Configurações Avançadas", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Noise",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Noise_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Length",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Usar texto de assistência", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Texto de assistência",
                        placeholder="Por que você está ignorando minha opinião? Imperdoável, irritante! Você deveria morrer.",
                        info="A voz e a emoção tendem a ser semelhantes à leitura deste texto. No entanto, a entonação e o tempo tendem a ser sacrificados.",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Intensidade do texto de assistência",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
                    )
                with gr.Accordion(label="Modelo Nulo", open=False):
                    with gr.Row():
                        null_models_count = gr.Number(
                            label="Número de modelos nulos", value=0, step=1
                        )
                    with gr.Column(variant="panel"):

                        @gr.render(inputs=[null_models_count])
                        def render_null_models(
                            null_models_count: int,
                        ):
                            global null_models_frame
                            null_models_frame = null_models_count
                            for i in range(null_models_count):
                                with gr.Row():
                                    null_model_index = gr.Number(
                                        value=i,
                                        key=f"null_model_index_{i}",
                                        visible=False,
                                    )
                                    null_model_name = gr.Dropdown(
                                        label="Lista de Modelos",
                                        choices=model_names,
                                        key=f"null_model_name_{i}",
                                        value=model_names[initial_id],
                                    )
                                    null_model_path = gr.Dropdown(
                                        label="Arquivo do Modelo",
                                        key=f"null_model_path_{i}",
                                        # FIXME: Gostaria de corrigir o problema de opções desaparecerem ao re-renderizar
                                        # Atualmente, ao re-renderizar, o valor é salvo mas as opções não, então as opções ficam vazias
                                        # Nesse momento, torna-se um valor que não está nas opções, então permitimos isso
                                        allow_custom_value=True,
                                    )
                                    null_voice_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_weights_{i}",
                                        label="Qualidade da voz",
                                    )
                                    null_voice_pitch_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_pitch_weights_{i}",
                                        label="Altura da voz",
                                    )
                                    null_speech_style_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_speech_style_weights_{i}",
                                        label="Estilo de fala",
                                    )
                                    null_tempo_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_tempo_weights_{i}",
                                        label="Tempo",
                                    )

                                    null_model_name.change(
                                        model_holder.update_model_files_for_gradio,
                                        inputs=[null_model_name],
                                        outputs=[null_model_path],
                                    )
                                    null_model_path.change(
                                        make_non_interactive, outputs=[tts_button]
                                    )
                                    # É muito direto/simples, gostaria de melhorar um pouco
                                    null_model_path.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_pitch_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_speech_style_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_tempo_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )

                    add_btn = gr.Button("Aumentar modelos nulos")
                    del_btn = gr.Button("Diminuir modelos nulos")
                    add_btn.click(
                        lambda x: x + 1,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )
                    del_btn.click(
                        lambda x: x - 1 if x > 0 else 0,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )

            with gr.Column():
                with gr.Accordion("Detalhes sobre o estilo", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["Escolher da predefinição", "Inserir arquivo de áudio"],
                    label="Método de especificação de estilo",
                    value="Escolher da predefinição",
                )
                style = gr.Dropdown(
                    label=f"Estilo ({DEFAULT_STYLE} é o estilo médio)",
                    choices=["Por favor, carregue o modelo"],
                    value="Por favor, carregue o modelo",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="Intensidade do estilo (reduza se a voz falhar)",
                )
                ref_audio_path = gr.Audio(
                    label="Áudio de referência", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "Síntese de Voz (Por favor, carregue o modelo)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="Informações")
                audio_output = gr.Audio(label="Resultado")
                with gr.Accordion("Exemplos de texto", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
                pitch_scale,
                intonation_scale,
                null_models,
                force_reload_model,
            ],
            outputs=[text_output, audio_output, tone, force_reload_model],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    import torch

    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(
        assets_root, device, torch_device_to_onnx_providers(device)
    )
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)
