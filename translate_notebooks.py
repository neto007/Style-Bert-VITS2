
import json
import os

def translate_colab_ipynb():
    file_path = "/home/machine/repository/google_audio/Style-Bert-VITS2/colab.ipynb"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    replacements = {
        "# Style-Bert-VITS2 (ver 2.7.0) のGoogle Colabでの学習": "# Treinamento do Style-Bert-VITS2 (ver 2.7.0) no Google Colab",
        "Google Colab上でStyle-Bert-VITS2の学習を行うことができます。": "Você pode treinar o Style-Bert-VITS2 no Google Colab.",
        "このnotebookでは、通常使用ではあなたのGoogle Driveにフォルダ`Style-Bert-VITS2`を作り、その内部での作業を行います。他のフォルダには触れません。": "Neste notebook, para uso normal, uma pasta `Style-Bert-VITS2` será criada no seu Google Drive e o trabalho será realizado dentro dela. Nenhuma outra pasta será tocada.",
        "Google Driveを使わない場合は、初期設定のところで適切なパスを指定してください。": "Se você não usar o Google Drive, especifique o caminho apropriado na configuração inicial.",
        "## 流れ": "## Fluxo",
        "### 学習を最初からやりたいとき": "### Quando quiser começar o treinamento do zero",
        "上から順に実行していけばいいです。音声合成に必要なファイルはGoogle Driveの`Style-Bert-VITS2/model_assets/`に保存されます。また、途中経過も`Style-Bert-VITS2/Data/`に保存されるので、学習を中断したり、途中から再開することもできます。": "Basta executar de cima para baixo. Os arquivos necessários para síntese de voz serão salvos em `Style-Bert-VITS2/model_assets/` no Google Drive. Além disso, o progresso intermediário também é salvo em `Style-Bert-VITS2/Data/`, para que você possa interromper o treinamento ou retomá-lo do meio.",
        "### 学習を途中から再開したいとき": "### Quando quiser retomar o treinamento do meio",
        "0と1を行い、3の前処理は飛ばして、4から始めてください。スタイル分け5は、学習が終わったら必要なら行ってください。": "Execute 0 e 1, pule o pré-processamento 3 e comece do 4. A divisão de estilo 5 deve ser feita após o término do treinamento, se necessário.",
        "## 0. 環境構築": "## 0. Configuração do Ambiente",
        "Style-Bert-VITS2の環境をcolab上に構築します。ランタイムがT4等のGPUバックエンドになっていることを確認し、実行してください。": "Construa o ambiente Style-Bert-VITS2 no Colab. Verifique se o runtime é um backend de GPU como T4 e execute.",
        "**注意**: このセルを実行した後に「セッションがクラッシュしました」「不明な理由により、セッションがクラッシュしました。」等の警告が出ますが、**無視してそのまま先へ**進んでください。（一度ランタイムを再起動させてnumpy<2を強制させるため `exit()` を呼んでいることからの措置です。）": "**Atenção**: Após executar esta célula, avisos como \"A sessão falhou\" ou \"A sessão falhou por motivo desconhecido\" aparecerão, mas **ignore e prossiga**. (Isso ocorre porque `exit()` é chamado para reiniciar o runtime e forçar numpy<2.)",
        "# Google driveを使う方はこちらを実行してください。": "# Execute isto se você usar o Google Drive.",
        "## 1. 初期設定": "## 1. Configuração Inicial",
        "学習とその結果を保存するディレクトリ名を指定します。": "Especifique o nome do diretório para salvar o treinamento e seus resultados.",
        "Google driveの場合はそのまま実行、カスタマイズしたい方は変更して実行してください。": "Se for Google Drive, execute como está; se quiser personalizar, altere e execute.",
        "# 作業ディレクトリを移動": "# Mover para o diretório de trabalho",
        "# 学習に必要なファイルや途中経過が保存されるディレクトリ": "# Diretório onde arquivos necessários para treinamento e progresso intermediário são salvos",
        "# 学習結果（音声合成に必要なファイルたち）が保存されるディレクトリ": "# Diretório onde os resultados do treinamento (arquivos necessários para síntese de voz) são salvos",
        "## 2. 学習に使うデータ準備": "## 2. Preparação de Dados para Treinamento",
        "すでに音声ファイル（1ファイル2-12秒程度）とその書き起こしデータがある場合は2.2を、ない場合は2.1を実行してください。": "Se você já tiver arquivos de áudio (cerca de 2-12 segundos por arquivo) e seus dados de transcrição, execute 2.2; caso contrário, execute 2.1.",
        "### 2.1 音声ファイルからのデータセットの作成（ある人はスキップ可）": "### 2.1 Criação de Dataset a partir de Arquivos de Áudio (pode pular se já tiver)",
        "音声ファイル（1ファイル2-12秒程度）とその書き起こしのデータセットを持っていない方は、（日本語の）音声ファイルのみから以下の手順でデータセットを作成することができます。Google drive上の`Style-Bert-VITS2/inputs/`フォルダに音声ファイル（wavやmp3等の通常の音声ファイル形式、1ファイルでも複数ファイルでも可）を置いて、下を実行すると、データセットが作られ、自動的に正しい場所へ配置されます。": "Se você não tiver um dataset de arquivos de áudio (cerca de 2-12 segundos por arquivo) e suas transcrições, pode criar um dataset a partir de arquivos de áudio (em japonês) seguindo os passos abaixo. Coloque os arquivos de áudio (formatos normais como wav ou mp3, um ou vários arquivos) na pasta `Style-Bert-VITS2/inputs/` no Google Drive e execute o comando abaixo para criar o dataset e colocá-lo automaticamente no local correto.",
        "**2024-06-02のVer 2.5以降**、`inputs/`フォルダにサブフォルダを2個以上作ってそこへ音声ファイルをスタイルに応じて振り分けて置くと、学習の際にサブディレクトリに応じたスタイルが自動的に作成されます。デフォルトスタイルのみでよい場合や手動でスタイルを後で作成する場合は`inputs/`直下へ入れれば大丈夫です。": "**A partir da Ver 2.5 de 02/06/2024**, se você criar 2 ou mais subpastas na pasta `inputs/` e distribuir os arquivos de áudio de acordo com o estilo, os estilos correspondentes aos subdiretórios serão criados automaticamente durante o treinamento. Se apenas o estilo padrão for suficiente ou se você for criar estilos manualmente depois, pode colocá-los diretamente em `inputs/`.",
        "# 元となる音声ファイル（wav形式）を入れるディレクトリ": "# Diretório para colocar arquivos de áudio de origem (formato wav)",
        "# モデル名（話者名）を入力": "# Digite o nome do modelo (nome do falante)",
        "# こういうふうに書き起こして欲しいという例文（句読点の入れ方・笑い方や固有名詞等）": "# Exemplo de frase como você quer que seja transcrito (pontuação, risadas, nomes próprios, etc.)",
        'initial_prompt = "こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！"': 'initial_prompt = "Olá. Tudo bem? Hehe, eu... estou bem!"',
        "成功したらそのまま3へ進んでください": "Se tiver sucesso, prossiga para o passo 3",
        "### 2.2 音声ファイルと書き起こしデータがすでにある場合": "### 2.2 Se você já tiver arquivos de áudio e dados de transcrição",
        "指示に従って適切にデータセットを配置してください。": "Coloque o dataset adequadamente seguindo as instruções.",
        "次のセルを実行して、学習データをいれるフォルダ（1で設定した`dataset_root`）を作成します。": "Execute a próxima célula para criar a pasta onde colocar os dados de treinamento (`dataset_root` configurado em 1).",
        "まず音声データと、書き起こしテキストを用意してください。": "Primeiro, prepare os dados de áudio e o texto de transcrição.",
        "それを次のように配置します。": "Organize-os da seguinte maneira:",
        "### 配置の仕方": "### Como organizar",
        "- 上のように配置すると、`style1/`と`style2/`フォルダの内部（直下以外も含む）に入っている音声ファイルたちから、自動的にデフォルトスタイルに加えて`style1`と`style2`というスタイルが作成されます": "- Se organizado como acima, além do estilo padrão, os estilos `style1` e `style2` serão criados automaticamente a partir dos arquivos de áudio dentro das pastas `style1/` e `style2/` (incluindo subpastas).",
        "- 特にスタイルを作る必要がない場合や、スタイル分類機能等でスタイルを作る場合は、`raw/`フォルダ直下に全てを配置してください。このように`raw/`のサブディレクトリの個数が0または1の場合は、スタイルはデフォルトスタイルのみが作成されます。": "- Se não precisar criar estilos específicos, ou se for criar estilos usando a função de classificação de estilo, coloque tudo diretamente na pasta `raw/`. Se o número de subdiretórios em `raw/` for 0 ou 1, apenas o estilo padrão será criado.",
        "- 音声ファイルのフォーマットはwav形式以外にもmp3等の多くの音声ファイルに対応しています": "- O formato do arquivo de áudio suporta muitos formatos como mp3 além do formato wav.",
        "### 書き起こしファイル`esd.list`": "### Arquivo de transcrição `esd.list`",
        "`Data/{モデルの名前}/esd.list` ファイルには、以下のフォーマットで各音声ファイルの情報を記述してください。": "No arquivo `Data/{nome_do_modelo}/esd.list`, descreva as informações de cada arquivo de áudio no seguinte formato:",
        "path/to/audio.wav(wavファイル以外でもこう書く)|{話者名}|{言語ID、ZHかJPかEN}|{書き起こしテキスト}": "path/to/audio.wav(mesmo se não for wav)|{nome_do_falante}|{ID_do_idioma, ZH, JP ou EN}|{texto_de_transcrição}",
        "- ここで、最初の`path/to/audio.wav`は、`raw/`からの相対パスです。つまり、`raw/foo.wav`の場合は`foo.wav`、`raw/style1/bar.wav`の場合は`style1/bar.wav`となります。": "- Aqui, o primeiro `path/to/audio.wav` é o caminho relativo a partir de `raw/`. Ou seja, para `raw/foo.wav` é `foo.wav`, e para `raw/style1/bar.wav` é `style1/bar.wav`.",
        "- 拡張子がwavでない場合でも、`esd.list`には`wav`と書いてください、つまり、`raw/bar.mp3`の場合でも`bar.wav`と書いてください。": "- Mesmo que a extensão não seja wav, escreva `wav` no `esd.list`. Por exemplo, para `raw/bar.mp3`, escreva `bar.wav`.",
        "例：": "Exemplo:",
        "foo.wav|hanako|JP|こんにちは、元気ですか？": "foo.wav|hanako|JP|Olá, como vai?",
        "bar.wav|taro|JP|はい、聞こえています……。何か用ですか？": "bar.wav|taro|JP|Sim, estou ouvindo... Precisa de algo?",
        "style1/baz.wav|hanako|JP|今日はいい天気ですね。": "style1/baz.wav|hanako|JP|O tempo está bom hoje, não é?",
        "style1/qux.wav|taro|JP|はい、そうですね。": "style1/qux.wav|taro|JP|Sim, é verdade.",
        "もちろん日本語話者の単一話者データセットでも構いません。": "Claro, um conjunto de dados de um único falante japonês também é aceitável.",
        "## 3. 学習の前処理": "## 3. Pré-processamento do Treinamento",
        "次に学習の前処理を行います。必要なパラメータをここで指定します。次のセルに設定等を入力して実行してください。「～～かどうか」は`True`もしくは`False`を指定してください。": "Em seguida, realizamos o pré-processamento do treinamento. Especifique os parâmetros necessários aqui. Insira as configurações na próxima célula e execute. Especifique `True` ou `False` para opções de \"sim/não\".",
        "# 上でつけたフォルダの名前`Data/{model_name}/`": "# Nome da pasta criada acima `Data/{model_name}/`",
        "# JP-Extra （日本語特化版）を使うかどうか。日本語の能力が向上する代わりに英語と中国語は使えなくなります。": "# Usar JP-Extra (versão especializada em japonês)? Melhora a capacidade em japonês, mas perde inglês e chinês.",
        "# 学習のバッチサイズ。VRAMのはみ出具合に応じて調整してください。": "# Tamanho do batch de treinamento. Ajuste de acordo com a VRAM disponível.",
        "# 学習のエポック数（データセットを合計何周するか）。": "# Número de épocas de treinamento (quantas vezes percorrer o dataset).",
        "# 100で多すぎるほどかもしれませんが、もっと多くやると質が上がるのかもしれません。": "# 100 pode ser demais, mas mais épocas podem melhorar a qualidade.",
        "# 保存頻度。何ステップごとにモデルを保存するか。分からなければデフォルトのままで。": "# Frequência de salvamento. A cada quantos passos salvar o modelo. Se não souber, deixe o padrão.",
        "# 音声ファイルの音量を正規化するかどうか": "# Normalizar o volume dos arquivos de áudio?",
        "# 音声ファイルの開始・終了にある無音区間を削除するかどうか": "# Remover silêncio no início e fim dos arquivos de áudio?",
        "# 読みのエラーが出た場合にどうするか。": "# O que fazer em caso de erro de leitura.",
        "# \"raise\"ならテキスト前処理が終わったら中断、\"skip\"なら読めない行は学習に使わない、\"use\"なら無理やり使う": "# \"raise\" interrompe após pré-processamento de texto, \"skip\" não usa linhas ilegíveis, \"use\" força o uso.",
        "上のセルが実行されたら、次のセルを実行して学習の前処理を行います。": "Após executar a célula acima, execute a próxima célula para realizar o pré-processamento do treinamento.",
        "## 4. 学習": "## 4. Treinamento",
        "前処理が正常に終わったら、学習を行います。次のセルを実行すると学習が始まります。": "Se o pré-processamento terminar normalmente, prossiga para o treinamento. Execute a próxima célula para iniciar.",
        "学習の結果は、上で指定した`save_every_steps`の間隔で、Google Driveの中の`Style-Bert-VITS2/Data/{モデルの名前}/model_assets/`フォルダに保存されます。": "Os resultados do treinamento serão salvos na pasta `Style-Bert-VITS2/Data/{nome_do_modelo}/model_assets/` no Google Drive no intervalo `save_every_steps` especificado acima.",
        "このフォルダをダウンロードし、ローカルのStyle-Bert-VITS2の`model_assets`フォルダに上書きすれば、学習結果を使うことができます。": "Baixe esta pasta e sobrescreva a pasta `model_assets` do seu Style-Bert-VITS2 local para usar os resultados do treinamento.",
        "# 上でつけたモデル名を入力。学習を途中からする場合はきちんとモデルが保存されているフォルダ名を入力。": "# Digite o nome do modelo definido acima. Se estiver retomando o treinamento, digite o nome da pasta onde o modelo está salvo.",
        "# 日本語特化版を「使う」場合": "# Se \"usar\" a versão especializada em japonês",
        "# 日本語特化版を「使わない」場合": "# Se \"não usar\" a versão especializada em japonês",
        "# 学習結果を試す・マージ・スタイル分けはこちらから": "# Testar resultados, mesclar e dividir estilos aqui",
        "# ONNX変換は、変換したいsafetensorsファイルを指定してこのセルを実行してください。": "# Para conversão ONNX, especifique o arquivo safetensors que deseja converter e execute esta célula.",
        "{モデルの名前}": "{nome_do_modelo}"
    }

    for cell in data["cells"]:
        if "source" in cell:
            new_source = []
            for line in cell["source"]:
                original_line = line
                for jp, pt in replacements.items():
                    if jp in line:
                        line = line.replace(jp, pt)
                new_source.append(line)
            cell["source"] = new_source

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

def translate_library_ipynb():
    file_path = "/home/machine/repository/google_audio/Style-Bert-VITS2/library.ipynb"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    replacements = {
        "# Style-Bert-VITS2ライブラリの使用例": "# Exemplo de uso da biblioteca Style-Bert-VITS2",
        "`pip install style-bert-vits2`を使った、jupyter notebookでの使用例です。Google colab等でも動きます。": "Este é um exemplo de uso em um notebook Jupyter usando `pip install style-bert-vits2`. Também funciona no Google Colab, etc.",
        "# PyTorch環境の構築（ない場合）": "# Construção do ambiente PyTorch (se não houver)",
        "# 参照: https://pytorch.org/get-started/locally/": "# Referência: https://pytorch.org/get-started/locally/",
        "# style-bert-vits2のインストール": "# Instalação do style-bert-vits2",
        "# BERTモデルをロード（ローカルに手動でダウンロードする必要はありません）": "# Carregar modelo BERT (não é necessário baixar manualmente para o local)",
        "# Hugging Faceから試しにデフォルトモデルをダウンロードしてみて、それを音声合成に使ってみる": "# Tente baixar o modelo padrão do Hugging Face e usá-lo para síntese de voz",
        "# model_assetsディレクトリにダウンロードされます": "# Será baixado no diretório model_assets",
        "# 上でダウンロードしたモデルファイルを指定して音声合成のテスト": "# Teste de síntese de voz especificando o arquivo de modelo baixado acima"
    }

    for cell in data["cells"]:
        if "source" in cell:
            new_source = []
            for line in cell["source"]:
                for jp, pt in replacements.items():
                    if jp in line:
                        line = line.replace(jp, pt)
                new_source.append(line)
            cell["source"] = new_source

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    translate_colab_ipynb()
    translate_library_ipynb()
