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


# Iniciar pyopenjtalk_worker
## O pyopenjtalk_worker é um servidor TCP socket, portanto iniciamos aqui
pyopenjtalk.initialize_worker()

# Para evitar consumo desnecessário de GPU VRAM durante o treinamento na Web UI, optamos por não pré-carregar o modelo BERT aqui
# As características BERT do dataset são extraídas previamente por bert_gen.py, portanto não é necessário carregar o modelo BERT durante o treinamento
# O pré-carregamento do modelo BERT ocorre dentro de TTSModelHolder.get_model_for_gradio() quando o botão "Carregar" é pressionado
# Durante o treinamento na Web UI, se o botão "Carregar" da aba de síntese de voz não for pressionado, o modelo BERT não será carregado na VRAM ao iniciar o treinamento

languages = [lang.value for lang in Languages]

initial_text = "Olá, prazer em conhecê-lo. Qual é o seu nome?"

examples = [
    [initial_text, "JP"],
    [
        """Você dizer isso me deixa muito feliz.
Você dizer isso me deixa muito irritado.
Você dizer isso me deixa muito surpreso.
Você dizer isso me deixa muito triste.""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった告白セリフ
        """Eu venho observando você há muito tempo. Seu sorriso, gentileza e força me cativaram.
Ao passar tempo como amigo, percebi que você se tornou cada vez mais especial.
Então, eu gosto de você! Se quiser, poderia namorar comigo?""",
        "JP",
    ],
    [  # 夏目漱石『吾輩は猫である』
        """Eu sou um gato. Ainda não tenho nome.
Não tenho ideia de onde nasci. Só lembro de chorar "miau" em um lugar escuro e úmido.
Foi aqui que vi um humano pela primeira vez. Mais tarde, ouvi dizer que ele era um estudante, a espécie mais agressiva entre os humanos.
Dizem que esses estudantes às vezes nos capturam e nos cozinham para comer.""",
        "JP",
    ],
    [  # 梶井基次郎『桜の樹の下には』
        """Há um cadáver sob a cerejeira! Isso é algo em que se pode acreditar.
Por que? Porque é incrível ver as flores de cerejeira florescerem tão lindamente. Eu não acreditava nessa beleza, então estava apreensivo.
Mas agora, finalmente entendi. Há um cadáver sob a cerejeira. Isso é algo em que se pode acreditar.""",
        "JP",
    ],
    [  # ChatGPTと考えた、感情を表すセリフ
        """Consegui! Tirei nota máxima no teste! Estou muito feliz!
Por que ignoram minha opinião? Não perdoo! Isso me irrita! Você deveria desaparecer.
Haha! Esse mangá é muito engraçado, veja isso, haha.
Sem você, fico tão triste que quase choro sozinho.""",
        "JP",
    ],
    [  # 上の丁寧語バージョン
        """Consegui! Tirei nota máxima no teste! Estou muito feliz!
Por que ignoram minha opinião? Não perdoo! Isso me irrita! Por favor, desapareça.
Haha! Esse mangá é muito engraçado, veja isso, haha.
Sem você, fico tão triste que quase choro sozinho.""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった音声合成の説明文章
        """Síntese de voz usa aprendizado de máquina para reproduzir a voz humana a partir de texto. Essa tecnologia analisa a estrutura da linguagem e gera áudio com base nisso.
Usando os últimos avanços da pesquisa, é possível gerar áudio mais natural e expressivo. Deep learning permite reproduzir nuances de timbre, incluindo emoções e sotaques.""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "Síntese de voz é a produção artificial de fala humana. Sistemas de computador para esse fim são chamados sintetizadores de voz e podem ser implementados via software ou hardware.",
        "ZH",
    ],
]

initial_md = """
- Os modelos padrão adicionados na Versão 2.5, [`koharune-ami`](https://huggingface.co/litagin/sbv2_koharune_ami) e [`amitaro`](https://huggingface.co/litagin/sbv2_amitaro), foram treinados usando corpus e gravações ao vivo disponíveis no [Amitaro Voice Material Workshop](https://amitaro.net/), com permissão prévia. Por favor, **leia os termos de uso** antes de utilizá-los.

- Ver 2.5のアップデート後に上記モデルをダウンロードするには、`Initialize.bat`をダブルクリックするか、手動でダウンロードして`model_assets`ディレクトリに配置してください。

- A **versão editor** adicionada na Versão 2.3 pode ser mais fácil de usar para síntese. Inicie com `Editor.bat` ou `python server_editor.py --inbrowser`.
"""

terms_of_use_md = """
## Pedido e Licença do Modelo Padrão

Consulte os termos de uso mais recentes [aqui](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md); eles são sempre aplicáveis.

Ao usar Style-Bert-VITS2, agradecemos se seguir as solicitações abaixo. Note que as solicitações anteriores à licença do modelo são apenas **pedidos** sem força legal e não constituem os termos de uso do Style-Bert-VITS2. Portanto, não conflitam com a [licença do repositório](https://github.com/litagin02/Style-Bert-VITS2#license), que permanece a única obrigação legal ao usar o código.

### O que não deve ser feito

Não queremos que o Style-Bert-VITS2 seja usado para os seguintes fins:

- Propósitos ilegais
- Propósitos políticos (proibidos pelo Bert-VITS2 original)
- Danificar outras pessoas
- Criação de deepfakes ou falsificação de identidade

### O que deve ser respeitado

- Ao usar Style-Bert-VITS2, verifique sempre os termos de uso e licenças dos modelos utilizados e siga-os se existirem.
- Ao usar o código-fonte, siga a [licença do repositório](https://github.com/litagin02/Style-Bert-VITS2#license).

A seguir estão as licenças dos modelos padrão.

### Corpus JVNV (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- O corpus [JVNV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) está licenciado sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja); adotamos essa licença.

### Koharune-ami / Amitaro

É obrigatório cumprir todas as regras do [Regulamento de Material Vocal da Amitaro](https://amitaro.net/voice/voice_rule/) e os [Termos de Uso das Gravações ao Vivo da Amitaro](https://amitaro.net/voice/livevoice/#index_id6). Em particular, siga estas diretrizes (permitidas tanto para uso comercial quanto não comercial):

#### Proibições

- Uso em obras ou aplicações com restrição de idade
- Conteúdo ligado a religiões novas, política ou esquemas de marketing multi‑nível
- Obras que difamam indivíduos, grupos ou nações
- Tratar áudio gerado como se fosse a voz real da Amitaro
- Tratar áudio gerado como se fosse a voz de outra pessoa que não a Amitaro

#### Créditos

Ao publicar áudio gerado (independentemente da mídia), inclua uma nota de crédito clara indicando que o modelo de voz foi baseado no material da **Amitaro Voice Material Workshop** (https://amitaro.net/).

Exemplos de crédito:
- `Modelo Style‑BertVITS2: Koharune‑ami, Amitaro Voice Material Workshop (https://amitaro.net/)`
- `Modelo Style‑BertVITS2: Amitaro, Amitaro Voice Material Workshop (https://amitaro.net/)`

#### Modelo de Mesclagem

Para a mesclagem de modelos, siga as [Perguntas Frequentes da Amitaro Voice Material Workshop](https://amitaro.net/voice/faq/#index_id17):
- Este modelo só pode ser mesclado com outro modelo se o detentor dos direitos da voz usada para criar esse outro modelo tiver dado permissão.
- Se as características da voz de Amitaro permanecerem (se a proporção da mesclagem for de 25% ou mais), o uso será restrito aos [Termos de Uso da Amitaro Voice Material Workshop](https://amitaro.net/voice/voice_rule/), e esses termos se aplicarão ao modelo resultante.
"""

how_to_md = """
下のように`model_assets`ディレクトリの中にモデルファイルたちを置いてください。
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
各モデルにはファイルたちが必要です：
- `config.json`：学習時の設定ファイル
- `*.safetensors`：学習済みモデルファイル（1つ以上が必要、複数可）
- `style_vectors.npy`：スタイルベクトルファイル

上2つは`Train.bat`による学習で自動的に正しい位置に保存されます。`style_vectors.npy`は`Style.bat`を実行して指示に従って生成してください。
"""

style_md = f"""
- プリセットまたは音声ファイルから読み上げの声音・感情・スタイルのようなものを制御できます。
- デフォルトの{DEFAULT_STYLE}でも、十分に読み上げる文に応じた感情で感情豊かに読み上げられます。このスタイル制御は、それを重み付きで上書きするような感じです。
- 強さを大きくしすぎると発音が変になったり声にならなかったりと崩壊することがあります。
- どのくらいに強さがいいかはモデルやスタイルによって異なるようです。
- 音声ファイルを入力する場合は、学習データと似た声音の話者（特に同じ性別）でないとよい効果が出ないかもしれません。
"""
voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]


def make_interactive():
    return gr.update(interactive=True, value="音声合成")


def make_non_interactive():
    return gr.update(interactive=False, value="音声合成（モデルをロードしてください）")


def gr_util(item):
    if item == "プリセットから選ぶ":
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
                wrong_tone_message = "Ajuste de acento atualmente suportado apenas para japonês."
            if line_split:
                logger.warning("Tone generation is not supported when splitting by lines.")
                wrong_tone_message = (
                    "Ajuste de acento não pode ser usado quando a geração é dividida por linhas."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"Ajuste de acento inválido: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
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
            return f"Error: Ajuste de acento inválido:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
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
            f"Nenhum modelo encontrado. Por favor, coloque os modelos em {model_holder.root_dir}."
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: Nenhum modelo encontrado. Por favor, coloque os modelos em {model_holder.root_dir}."
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
                            label="Lista de modelos",
                            choices=model_names,
                            value=model_names[initial_id],
                        )
                        model_path = gr.Dropdown(
                            label="Arquivo do modelo",
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
                    label="Pitch (qualidade pode degradar fora de 1)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="Entonação (qualidade pode degradar fora de 1)",
                )

                line_split = gr.Checkbox(
                    label="Dividir por quebras de linha (geralmente melhora a emoção)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="Duração do silêncio entre linhas (segundos)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="Ajuste de acento (0=baixo, 1=alto)",
                    info="Só funciona quando não há divisão por linhas. Não é universal.",
                )
                use_tone = gr.Checkbox(label="Usar ajuste de acento", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="PT", label="Idioma")
                speaker = gr.Dropdown(label="Locutor")
                with gr.Accordion(label="Detalhes", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="Relação SDP",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Ruído",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Ruído_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Duração",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Usar texto de assistência", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Texto de assistência",
                        placeholder="Por que ignoram minha opinião? Não perdoo, estou irritado! Você deveria desaparecer.",
                        info="A leitura desse texto tende a produzir voz e emoção semelhantes. Pode sacrificar entonação e ritmo.",
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
                                        label="Lista de modelos",
                                        choices=model_names,
                                        key=f"null_model_name_{i}",
                                        value=model_names[initial_id],
                                    )
                                    null_model_path = gr.Dropdown(
                                        label="Arquivo do modelo",
                                        key=f"null_model_path_{i}",
                                        # FIXME: As opções desaparecem ao re-renderizar; ainda assim permitimos valores personalizados
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
                                        label="Tom de voz",
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
                                    # 愚直すぎるのでもう少しなんとかしたい
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

                    add_btn = gr.Button("Adicionar modelo nulo")
                    del_btn = gr.Button("Remover modelo nulo")
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
                with gr.Accordion("Detalhes do estilo", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["プリセットから選ぶ", "音声ファイルを入力"],
                    label="Método de especificação de estilo",
                    value="プリセットから選ぶ",
                )
                style = gr.Dropdown(
                    label=f"Estilo (padrão {DEFAULT_STYLE})",
                    choices=["Carregue o modelo primeiro"],
                    value="Carregue o modelo primeiro",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="Intensidade do estilo (reduza se a voz distorcer)",
                )
                ref_audio_path = gr.Audio(
                    label="Áudio de referência", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "Síntese de voz (carregue o modelo primeiro)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="Informação")
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
