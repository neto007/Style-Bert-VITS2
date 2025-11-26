# Style-Bert-VITS2

**Por favor, leia os [Termos de Uso e Avisos](/docs/TERMS_OF_USE.md) antes de usar.**

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/e853f9a2-db4a-4202-a1dd-56ded3c562a0

You can install via `pip install style-bert-vits2` (inference only), see [library.ipynb](/library.ipynb) for example usage.

- **Vídeo tutorial explicativo** [YouTube](https://youtu.be/aTUSzgDl1iY) [Vídeo NicoNico](https://www.nicovideo.jp/watch/sm43391524)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- [**Perguntas Frequentes** (FAQ)](/docs/FAQ.md)
- [🤗 オンラインデモはこちらから](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)
- [Artigo explicativo no Zenn](https://zenn.dev/litagin/articles/034819a5256ff4)

- [**Página de lançamentos**](https://github.com/litagin02/Style-Bert-VITS2/releases/)、[Histórico de atualizações](/docs/CHANGELOG.md)
  - 2025-08-24: Versão 2.7.0: Adição de GUI para conversão ONNX e integração com bibliotecas externas como [Aivis Project](https://aivis-project.com/), além de incluir o modelo de reconhecimento de voz `litagin/anime-whisper`.
  - 2024-09-09: Versão 2.6.1: Correção de bugs que impediam o treinamento adequado no Google Colab.
  - 2024-06-16: Versão 2.6.0 (adição de mesclagem de diferenças de modelo, mesclagem ponderada e mesclagem de modelo nulo; veja [este artigo](https://zenn.dev/litagin/articles/1297b1dc7bdc79) para detalhes de uso).
  - 2024-06-14: Versão 2.5.1 (apenas mudou os termos de uso para um aviso).
  - 2024-06-02: Versão 2.5.0 (**Adição dos [Termos de Uso](/docs/TERMS_OF_USE.md)**, geração de estilos a partir de organização de pastas, inclusão dos modelos 小春音アミ・あみたろ, e aceleração da instalação).
  - 2024-03-16: Versão 2.4.1 (**alteração do método de instalação via arquivos .bat**).
  - 2024-03-15: Versão 2.4.0 (refatoração em larga escala, várias melhorias e modularização como biblioteca).
  - 2024-02-26: Versão 2.3 (funcionalidade de dicionário e editor).
  - 2024-02-09: ver 2.2
  - 2024-02-07: ver 2.1
  - 2024-02-03: ver 2.0 (JP-Extra)
  - 2024-01-09: ver 1.3
  - 2023-12-31: ver 1.2
  - 2023-12-29: ver 1.1
  - 2023-12-27: ver 1.0

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 and Japanese-Extra, so many thanks to the original author!

**Visão geral**

- Gera áudio emocionalmente rico a partir do texto inserido, baseado no Bert-VITS2 v2.1 e Japanese-Extra, permitindo controle livre de emoções e estilos de fala, inclusive intensidade.
- Mesmo sem Git ou Python (usuários Windows), a instalação é simples e o treinamento é possível (grande parte baseada no [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2/)). O treinamento também é suportado no Google Colab: [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- Para uso apenas de síntese de voz, funciona em CPU mesmo sem GPU.
- Para síntese de voz somente, pode ser instalado como biblioteca Python via `pip install style-bert-vits2`. Veja o exemplo em [library.ipynb](/library.ipynb).
- Também inclui um servidor API para integração com outros sistemas (contribuição de [@darai0512](https://github.com/darai0512)).
- Originalmente, Bert-VITS2 destaca-se por ler "textos alegres de forma alegre e textos tristes de forma triste", permitindo gerar áudio rico em emoções mesmo com estilos padrão.


## Como usar

- Consulte [aqui](/docs/CLI.md) para instruções de uso via CLI.
- Veja também as [Perguntas Frequentes](/docs/FAQ.md).

### Ambiente de execução

Testado em Windows Command Prompt, WSL2 e Linux (Ubuntu Desktop) para UI e servidor API (ajuste caminhos relativos no WSL). Sem GPU Nvidia, o treinamento não funciona, mas síntese e mesclagem de áudio ainda são possíveis.

### インストール

Para instalar via pip como biblioteca Python e exemplos de uso, veja [library.ipynb](/library.ipynb).

#### Para quem não está familiarizado com Git ou Python

Baseado em Windows.

1. Baixe o [arquivo zip](https://github.com/litagin02/Style-Bert-VITS2/releases/latest/download/sbv2.zip) para um diretório sem caracteres japoneses ou espaços e extraia.
  - Se possuir GPU, clique duas vezes em `Install-Style-Bert-VITS2.bat`.
  - Se não houver GPU, clique duas vezes em `Install-Style-Bert-VITS2-CPU.bat`. A versão CPU não permite treinamento, mas suporta síntese e mesclagem.
2. Aguarde enquanto o ambiente necessário é instalado automaticamente.
3. Quando o editor de síntese de voz iniciar automaticamente, a instalação foi bem-sucedida. O modelo padrão já está baixado, pronto para uso.

Para atualizar, clique duas vezes em `Update-Style-Bert-VITS2.bat`.

Observação: ao atualizar de versões anteriores à **2.4.1** (antes de 2024-03-16), será necessário remover tudo e reinstalar. Consulte o procedimento em [CHANGELOG.md](/docs/CHANGELOG.md).

#### Para quem tem experiência com Git e Python

Recomendamos usar o gerenciador de pacotes [uv](https://github.com/astral-sh/uv), que é mais rápido que o pip, para criar ambientes virtuais Python.
(Se preferir, o pip tradicional também funciona.)

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv venv venv
venv\Scripts\activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py  # Baixa os modelos necessários e o modelo TTS padrão
```
Não se esqueça do passo final.

### Síntese de voz

O editor de síntese de voz inicia ao clicar duas vezes em `Editor.bat` ou executar `python server_editor.py --inbrowser` (use `--device cpu` para modo CPU). Permite ajustar cada linha, salvar, carregar e editar dicionários.
Mesmo sem treinamento, o modelo padrão baixado na instalação pode ser usado.

A parte do editor está em um [repositório separado](https://github.com/litagin02/Style-Bert-VITS2-Editor).

Para versões anteriores à 2.2, a WebUI de síntese de voz inicia ao clicar em `App.bat` ou executar `python app.py`. Também é possível abrir a aba de síntese única via `Inference.bat`.

A estrutura dos arquivos de modelo necessários para síntese de voz é a seguinte (não requer posicionamento manual).
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
Para inferência, são necessários `config.json`, arquivos `*.safetensors` e `style_vectors.npy`. Ao compartilhar um modelo, inclua esses três arquivos.


O treinamento requer vários arquivos de áudio de cerca de 2 a 14 segundos e seus dados de transcrição.

- Se você já tiver arquivos de áudio divididos e dados de transcrição de um corpus existente, etc., poderá usá-los como estão (modificando o arquivo de transcrição, se necessário). Consulte "WebUI de Treinamento" abaixo.
- Caso contrário, se você tiver apenas arquivos de áudio (de qualquer duração), incluímos uma ferramenta para criar um conjunto de dados a partir deles para uso imediato no treinamento.

#### Criação de Dataset

- Na guia "Criação de Dataset" da WebUI, aberta clicando duas vezes em `App.bat` ou executando `python app.py`, você pode fatiar arquivos de áudio em comprimentos apropriados e transcrevê-los automaticamente. Alternativamente, clicar duas vezes em `Dataset.bat` abrirá essa guia isoladamente.
- Após seguir as instruções, você pode prosseguir diretamente para o treinamento na guia "Treinamento" abaixo.

#### WebUI de Treinamento

- Siga as instruções na guia "Treinamento" da WebUI, aberta clicando duas vezes em `App.bat` ou executando `python app.py`. Alternativamente, clicar duas vezes em `Train.bat` abrirá essa guia isoladamente.

### Geração de Estilo

- Por padrão, além do estilo padrão "Neutral", estilos correspondentes à divisão de pastas na pasta de treinamento são gerados.
- Isto é para quem deseja criar estilos manualmente por outros métodos.
- Você pode gerar estilos usando arquivos de áudio na guia "Criação de Estilo" da WebUI, aberta clicando duas vezes em `App.bat` ou executando `python app.py`. Alternativamente, clicar duas vezes em `StyleVectors.bat` abrirá essa guia isoladamente.
- Como é independente do treinamento, pode ser feito durante o treinamento ou refeito várias vezes após o término do treinamento (o pré-processamento deve estar concluído).

### API Server

Executar `python server_fastapi.py` no ambiente construído iniciará o servidor API.
As especificações da API podem ser verificadas em `/docs` após a inicialização.

- O limite de caracteres de entrada é 100 por padrão. Isso pode ser alterado em `server.limit` no `config.yml`.
- Por padrão, as configurações CORS permitem todos os domínios. Sempre que possível, altere o valor de `server.origins` no `config.yml` para restringir a domínios confiáveis (apagar a chave desativará as configurações CORS).

Além disso, o servidor API do editor de síntese de voz é iniciado com `python server_editor.py`. Mas ainda não está muito bem mantido. Atualmente, apenas a API mínima necessária do [repositório do editor](https://github.com/litagin02/Style-Bert-VITS2-Editor) está implementada.

Para implantação web do editor de síntese de voz, consulte [este Dockerfile](Dockerfile.deploy).

### マージ

Você pode criar um novo modelo misturando dois modelos em termos de "qualidade de voz", "tom de voz", "expressão emocional" e "tempo", ou realizar operações como "adicionar a diferença entre dois outros modelos a um modelo".
Você pode mesclar dois modelos selecionando-os na guia "Mesclagem" da WebUI, que pode ser aberta clicando duas vezes em `App.bat` ou executando `python app.py`. Alternativamente, clicar duas vezes em `Merge.bat` abrirá essa guia isoladamente.

### Conversão ONNX

Você pode converter arquivos safetensors treinados para o formato ONNX na guia "Conversão ONNX" ou usando `ConvertONNX.bat`. Isso é útil quando arquivos no formato ONNX são necessários para bibliotecas externas. Por exemplo, no [Aivis Project](https://aivis-project.com/), você pode usar o [AIVM Generator](https://aivm-generator.aivis-project.com/) para criar modelos para o Aivis Speech a partir de arquivos safetensors e ONNX.

### Avaliação de Naturalidade

Como "um" indicador de qual número de passos é melhor entre os resultados do treinamento, preparamos um script que usa [SpeechMOS](https://github.com/tarepan/SpeechMOS):
```bash
python speech_mos.py -m <model_name>
```
A avaliação de naturalidade para cada passo é exibida, e os resultados são salvos em `mos_{model_name}.csv` e `mos_{model_name}.png` na pasta `mos_results`. Se você quiser mudar o texto a ser lido, modifique o arquivo e ajuste-o você mesmo. Além disso, esta é apenas uma avaliação baseada em critérios que não consideram sotaque, expressão emocional ou entonação, servindo apenas como um guia, então acho que é melhor selecionar ouvindo a leitura real.

## Relação com Bert-VITS2

Basicamente, é apenas uma ligeira modificação da estrutura do modelo Bert-VITS2. Tanto o [modelo pré-treinado antigo](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) quanto o [modelo pré-treinado JP-Extra](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) usam essencialmente o mesmo que Bert-VITS2 v2.1 ou JP-Extra (com pesos desnecessários removidos e convertidos para safetensors).

Especificamente, os seguintes pontos são diferentes:

- Como o [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2), é fácil de usar mesmo para quem não conhece Python ou Git.
- Modelo de embedding de emoção alterado (para [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) de 256 dimensões; mais um embedding para identificação de falante do que embedding de emoção)
- A quantização vetorial também foi removida do embedding de emoção, tornando-se uma simples camada totalmente conectada.
- Ao criar o arquivo de vetor de estilo `style_vectors.npy`, você pode gerar voz usando esse estilo enquanto especifica continuamente a intensidade do efeito.
- Várias WebUIs criadas
- Suporte para treinamento em bf16
- Suporte ao formato safetensors, usando safetensors por padrão
- Outras pequenas correções de bugs e refatoração


## References
In addition to the original reference (written below), I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The pretrained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and [JP-Extra version](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) is essentially taken from [the original base model of Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) and [JP-Extra pretrained model of Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra), so all the credits go to the original author ([Fish Audio](https://github.com/fishaudio)):


In addition, [text/user_dict/](text/user_dict) module is based on the following repositories:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
and the license of this module is LGPL v3.

## LICENSE

This repository is licensed under the GNU Affero General Public License v3.0, the same as the original Bert-VITS2 repository. For more details, see [LICENSE](LICENSE).

In addition, [text/user_dict/](text/user_dict) module is licensed under the GNU Lesser General Public License v3.0, inherited from the original VOICEVOX engine repository. For more details, see [LGPL_LICENSE](LGPL_LICENSE).



Below is the original README.md.
---

<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Bert-VITS2

VITS2 Backbone with multilingual bert

For quick guide, please refer to `webui_preprocess.py`.

Para um tutorial simples, consulte `webui_preprocess.py`.

## Observe que a ideia central deste projeto vem de [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS), um excelente projeto de TTS
## A demonstração do MassTTS está em [ai版峰哥锐评峰哥本人,并找回了在金三角失落的腰子](https://www.bilibili.com/video/BV1w24y1c7z9)

[//]: # (## 本项目与[PlayVoice/vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41; 没有任何关系)

[//]: # ()
[//]: # (本仓库来源于之前朋友分享了ai峰哥的视频，本人被其中的效果惊艳，在自己尝试MassTTS以后发现fs在音质方面与vits有一定差距，并且training的pipeline比vits更复杂，因此按照其思路将bert)

## Viajantes/Pioneiros/Capitães/Doutores/Senseis/Witchers/MiaoMiaoLu/Vs experientes devem consultar o código e aprender a treinar por conta própria.

### É estritamente proibido usar este projeto para qualquer finalidade que viole a Constituição, o Código Penal, a Lei de Punição da Administração de Segurança Pública e o Código Civil da República Popular da China.
### É estritamente proibido o uso para quaisquer fins políticos.
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
#### QQ Group：815818430
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [emotional-vits](https://github.com/innnky/emotional-vits)
+ [fish-speech](https://github.com/fishaudio/fish-speech)
+ [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)
## Obrigado a todos os colaboradores por seus esforços
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>

[//]: # (# 本项目所有代码引用均已写明，bert部分代码思路来源于[AI峰哥]&#40;https://www.bilibili.com/video/BV1w24y1c7z9&#41;，与[vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41;无任何关系。欢迎各位查阅代码。同时，我们也对该开发者的[碰瓷，乃至开盒开发者的行为]&#40;https://www.bilibili.com/read/cv27101514/&#41;表示强烈谴责。)
