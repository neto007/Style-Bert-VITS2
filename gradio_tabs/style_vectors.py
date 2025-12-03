"""
TODO:
Os imports são pesados, deixando a WebUI lenta. Gostaria de resolver isso.
"""

import json
import shutil
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from umap import UMAP

from config import get_path_config
from default_style import save_styles_by_dirs
from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger


path_config = get_path_config()
dataset_root = path_config.dataset_root
assets_root = path_config.assets_root

MAX_CLUSTER_NUM = 10
MAX_AUDIO_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
umap = UMAP(n_components=2, random_state=42, metric="cosine", n_jobs=1, min_dist=0.0)

wav_files: list[Path] = []
x = np.array([])
x_reduced = None
y_pred = np.array([])
mean = np.array([])
centroids = []


def load(model_name: str, reduction_method: str):
    global wav_files, x, x_reduced, mean
    wavs_dir = dataset_root / model_name / "wavs"
    style_vector_files = [f for f in wavs_dir.rglob("*.npy") if f.is_file()]
    # foo.wav.npy -> foo.wav
    wav_files = [f.with_suffix("") for f in style_vector_files]
    logger.info(f"Found {len(style_vector_files)} style vectors in {wavs_dir}")
    style_vectors = [np.load(f) for f in style_vector_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)
    if reduction_method == "t-SNE":
        x_reduced = tsne.fit_transform(x)
    elif reduction_method == "UMAP":
        x_reduced = umap.fit_transform(x)
    else:
        raise ValueError("Invalid reduction method")
    x_reduced = np.asarray(x_reduced)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids, x_reduced, y_pred
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after reduction":
        assert x_reduced is not None
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_reduced)
    elif method == "Agglomerative after reduction":
        assert x_reduced is not None
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_reduced)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def do_dbscan(eps=2.5, min_samples=15):
    global centroids, x_reduced, y_pred
    model = DBSCAN(eps=eps, min_samples=min_samples)
    assert x_reduced is not None
    y_pred = model.fit_predict(x_reduced)
    n_clusters = max(y_pred) + 1
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))
    return y_pred, centroids


def representative_wav_files(cluster_id, num_files=1):
    # Procurar medoide relacionado ao cluster_index em y_pred
    cluster_indices = np.where(y_pred == cluster_id)[0]
    cluster_vectors = x[cluster_indices]
    # Calcular distâncias entre todos os vetores no cluster
    distances = pdist(cluster_vectors)
    distance_matrix = squareform(distances)

    # Calcular distância média de cada vetor para todos os outros vetores
    mean_distances = distance_matrix.mean(axis=1)

    # Obter num_files índices em ordem de menor distância média
    closest_indices = np.argsort(mean_distances)[:num_files]

    return cluster_indices[closest_indices]


def do_dbscan_gradio(eps=2.5, min_samples=15):
    global x_reduced, centroids

    y_pred, centroids = do_dbscan(eps, min_samples)

    assert x_reduced is not None

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(max(y_pred) + 1):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    # Noise cluster (-1) is black
    plt.scatter(
        x_reduced[y_pred == -1, 0],
        x_reduced[y_pred == -1, 1],
        color="black",
        label="Noise",
    )
    plt.legend()

    n_clusters = int(max(y_pred) + 1)

    if n_clusters > MAX_CLUSTER_NUM:
        # raise ValueError(f"The number of clusters is too large: {n_clusters}")
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"Número de clusters muito alto, tente alterar os parâmetros: {n_clusters}",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    elif n_clusters == 0:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            "O número de clusters é 0. Tente alterar os parâmetros.",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    return [plt, gr.Slider(maximum=n_clusters, value=1), n_clusters] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def representative_wav_files_gradio(cluster_id, num_files=1):
    cluster_id = cluster_id - 1  # Converter de 1-indexado (UI) para 0-indexado
    closest_indices = representative_wav_files(cluster_id, num_files)
    actual_num_files = len(closest_indices)  # Para quando há poucos arquivos
    return [
        gr.Audio(wav_files[i], visible=True, label=str(wav_files[i]))
        for i in closest_indices
    ] + [gr.update(visible=False)] * (MAX_AUDIO_NUM - actual_num_files)


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_reduced, centroids
    y_pred, centroids = do_clustering(n_clusters, method)

    assert x_reduced is not None
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    plt.legend()

    return [plt, gr.Slider(maximum=n_clusters, value=1)] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def save_style_vectors_from_clustering(model_name: str, style_names_str: str):
    """Salvar center e centroids"""
    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)
    logger.success(f"Saved style vectors to {style_vector_path}")

    # Atualizar config.json
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} não existe."
    style_names = [name.strip() for name in style_names_str.split(",")]
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(style_name_list) != len(centroids) + 1:
        return f"O número de estilos não corresponde. Verifique se está corretamente dividido em {len(centroids)} por vírgulas: {style_names_str}"
    if len(set(style_names)) != len(style_names):
        return "Os nomes dos estilos estão duplicados."

    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.success(f"Updated {config_path}")
    return f"Sucesso!\nSalvo em {style_vector_path} e {config_path} atualizado."


def save_style_vectors_from_files(
    model_name: str, audio_files_str: str, style_names_str: str
):
    """Criar e salvar vetores de estilo a partir de arquivos de áudio"""
    global mean
    if len(x) == 0:
        return "Erro: Por favor, carregue os vetores de estilo."
    mean = np.mean(x, axis=0)

    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [name.strip() for name in audio_files_str.split(",")]
    style_names = [name.strip() for name in style_names_str.split(",")]
    if len(audio_files) != len(style_names):
        return f"O número de arquivos de áudio e nomes de estilo não corresponde. Verifique se estão corretamente divididos por vírgulas em {len(style_names)}: {audio_files_str} e {style_names_str}"
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(set(style_names)) != len(style_names):
        return "Os nomes dos estilos estão duplicados."
    style_vectors = [mean]

    wavs_dir = dataset_root / model_name / "wavs"
    for audio_file in audio_files:
        path = wavs_dir / audio_file
        if not path.exists():
            return f"{path} não existe."
        style_vectors.append(np.load(f"{path}.npy"))
    style_vectors = np.stack(style_vectors)
    assert len(style_name_list) == len(style_vectors)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)

    # Atualizar config.json
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} não existe."
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"Sucesso!\nSalvo em {style_vector_path} e {config_path} atualizado."


def save_style_vectors_by_dirs(model_name: str, input_dir: str):
    if model_name == "":
        return "Erro: Por favor, insira o nome do modelo."
    if input_dir == "":
        return "Erro: Por favor, insira o diretório contendo os arquivos de áudio."
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists():
        return f"Erro: {input_dir} não existe."

    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    from tqdm import tqdm

    from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
    from style_gen import save_style_vector

    # First generate style vectors for each audio file

    audio_dir = Path(input_dir)
    audio_suffixes = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    audio_files = [f for f in audio_dir.rglob("*") if f.suffix in audio_suffixes]

    def process(file: Path):
        # f: `test.wav` -> search `test.wav.npy`
        if (file.with_name(file.name + ".npy")).exists():
            return file, None
        try:
            save_style_vector(str(file))
        except Exception as e:
            return file, e
        return file, None

    with ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
        _ = list(
            tqdm(
                executor.map(
                    process,
                    audio_files,
                ),
                total=len(audio_files),
                file=SAFE_STDOUT,
                desc="Gerando vetores de estilo",
                dynamic_ncols=True,
            )
        )

    result_dir = assets_root / model_name
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} não existe."
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    save_styles_by_dirs(
        wav_dir=audio_dir,
        output_dir=result_dir,
        config_path=config_path,
        config_output_path=config_path,
    )
    return f"Sucesso!\nVetores de estilo salvos em {result_dir}."


how_to_md = """
Crie um arquivo de vetor de estilo (`style_vectors.npy`) para o modelo.
Isso é necessário para controlar o estilo (emoção, tom, etc.) durante a síntese de voz.

Existem três métodos:
1. **Método 0**: Divida os arquivos de áudio em subpastas para cada estilo e calcule a média.
2. **Método 1**: Agrupe automaticamente os arquivos de áudio e calcule a média de cada cluster.
3. **Método 2**: Selecione manualmente os arquivos de áudio representativos para cada estilo.
"""

method0 = """
### Método 0: Criar vetores de estilo por subpasta
Coloque os arquivos de áudio em subpastas para cada estilo dentro da pasta `inputs`.
O nome da subpasta será o nome do estilo.

Exemplo de estrutura de diretório:
```
inputs
├── Neutral
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
├── Happy
│   ├── 101.wav
│   ├── 102.wav
│   └── ...
├── Sad
│   ├── 201.wav
│   ├── ...
└── ...
```
"""

method1 = """
### Método 1: Classificação automática de estilo (Clustering)
Analisa todos os arquivos de áudio na pasta e os classifica automaticamente em estilos semelhantes.
Você pode ajustar o número de estilos e o algoritmo de clustering.
"""

dbscan_md = """
### DBSCAN
Um algoritmo de clustering baseado em densidade.
- `eps`: A distância máxima entre dois pontos para serem considerados vizinhos.
- `min_samples`: O número mínimo de pontos necessários para formar um cluster.
"""


def create_style_vectors_app():
    with gr.Blocks(theme=GRADIO_THEME) as app:
        with gr.Accordion("Como usar", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(label="Nome do Modelo")
        with gr.Tab("Método 0: Por Subpasta"):
            gr.Markdown(method0)
            input_dir = gr.Textbox(
                label="Pasta contendo áudio",
                value="inputs",
                info="Por favor, salve os arquivos de áudio divididos em subpastas para cada estilo.",
            )
            method0_btn = gr.Button("Criar Vetores de Estilo", variant="primary")
            method0_info = gr.Textbox(label="Resultado")
            method0_btn.click(
                save_style_vectors_by_dirs,
                inputs=[model_name, input_dir],
                outputs=[method0_info],
            )
        with gr.Tab("Outros Métodos"):
            gr.Markdown("## Visualização e Classificação Automática")
            with gr.Row():
                reduction_method = gr.Radio(
                    choices=["UMAP", "t-SNE"],
                    label="Método de Redução de Dimensionalidade",
                    info="v 1.3 ou anterior era t-SNE, mas UMAP pode ser melhor.",
                    value="UMAP",
                )
                load_button = gr.Button("Carregar Vetores de Estilo", variant="primary")
            output = gr.Plot(label="Visualização de Estilos de Áudio")
            load_button.click(
                load, inputs=[model_name, reduction_method], outputs=[output]
            )
            with gr.Tab("Método 1: Classificação Automática"):
                gr.Markdown(method1)
                with gr.Tab("Classificação de Estilo 1"):
                    n_clusters = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=4,
                        label="Número de estilos a criar (excluindo o estilo médio)",
                        info="Tente ajustar o número de estilos observando a figura acima.",
                    )
                    c_method = gr.Radio(
                        choices=[
                            "Agglomerative after reduction",
                            "KMeans after reduction",
                            "Agglomerative",
                            "KMeans",
                        ],
                        label="Algoritmo",
                        info="Selecione o algoritmo de classificação (clustering). Tente vários.",
                        value="Agglomerative after reduction",
                    )
                    c_button = gr.Button("Executar Classificação de Estilo")
                with gr.Tab("Classificação de Estilo 2: DBSCAN"):
                    gr.Markdown(dbscan_md)
                    eps = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        step=0.01,
                        value=0.3,
                        label="eps",
                    )
                    min_samples = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=15,
                        label="min_samples",
                    )
                    with gr.Row():
                        dbscan_button = gr.Button("Executar Classificação de Estilo")
                        num_styles_result = gr.Textbox(label="Número de Estilos")
                gr.Markdown("Resultado da Classificação de Estilo")
                gr.Markdown(
                    "Nota: Como estamos reduzindo de 256 dimensões para 2 dimensões, a relação de posição dos vetores não é exata."
                )
                with gr.Row():
                    gr_plot = gr.Plot()
                    with gr.Column():
                        with gr.Row():
                            cluster_index = gr.Slider(
                                minimum=1,
                                maximum=MAX_CLUSTER_NUM,
                                step=1,
                                value=1,
                                label="Número do estilo",
                                info="Exibe o áudio representativo do estilo selecionado.",
                            )
                            num_files = gr.Slider(
                                minimum=1,
                                maximum=MAX_AUDIO_NUM,
                                step=1,
                                value=5,
                                label="Quantos áudios representativos exibir",
                            )
                            get_audios_button = gr.Button("Obter áudios representativos")
                        with gr.Row():
                            audio_list = []
                            for i in range(MAX_AUDIO_NUM):
                                audio_list.append(
                                    gr.Audio(visible=False, show_label=True)
                                )
                    c_button.click(
                        do_clustering_gradio,
                        inputs=[n_clusters, c_method],
                        outputs=[gr_plot, cluster_index] + audio_list,
                    )
                    dbscan_button.click(
                        do_dbscan_gradio,
                        inputs=[eps, min_samples],
                        outputs=[gr_plot, cluster_index, num_styles_result]
                        + audio_list,
                    )
                    get_audios_button.click(
                        representative_wav_files_gradio,
                        inputs=[cluster_index, num_files],
                        outputs=audio_list,
                    )
                gr.Markdown("Se o resultado parecer bom, salve-o.")
                style_names = gr.Textbox(
                    "Angry, Sad, Happy",
                    label="Nomes dos Estilos",
                    info=f"Insira os nomes dos estilos separados por vírgula (Português aceito). Ex: `Angry, Sad, Happy` ou `Raiva, Tristeza, Alegria`. O áudio médio é salvo automaticamente como {DEFAULT_STYLE}.",
                )
                with gr.Row():
                    save_button1 = gr.Button(
                        "Salvar Vetores de Estilo", variant="primary"
                    )
                    info2 = gr.Textbox(label="Resultado do Salvamento")

                save_button1.click(
                    save_style_vectors_from_clustering,
                    inputs=[model_name, style_names],
                    outputs=[info2],
                )
            with gr.Tab("Método 2: Seleção Manual"):
                gr.Markdown(
                    "Insira os nomes dos arquivos de áudio representativos de cada estilo separados por vírgula na caixa de texto abaixo, e os nomes dos estilos correspondentes separados por vírgula ao lado."
                )
                gr.Markdown("Exemplo: `angry.wav, sad.wav, happy.wav` e `Angry, Sad, Happy`")
                gr.Markdown(
                    f"Nota: O estilo {DEFAULT_STYLE} é salvo automaticamente, não especifique um estilo com esse nome manualmente."
                )
                with gr.Row():
                    audio_files_text = gr.Textbox(
                        label="Nome do arquivo de áudio",
                        placeholder="angry.wav, sad.wav, happy.wav",
                    )
                    style_names_text = gr.Textbox(
                        label="Nome do estilo", placeholder="Angry, Sad, Happy"
                    )
                with gr.Row():
                    save_button2 = gr.Button(
                        "Salvar Vetores de Estilo", variant="primary"
                    )
                    info2 = gr.Textbox(label="Resultado do Salvamento")
                    save_button2.click(
                        save_style_vectors_from_files,
                        inputs=[model_name, audio_files_text, style_names_text],
                        outputs=[info2],
                    )

    return app


if __name__ == "__main__":
    app = create_style_vectors_app()
    app.launch(inbrowser=True)
