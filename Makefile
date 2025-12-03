.PHONY: install-system install-deps run clean help

help:
	@echo "Comandos disponíveis:"
	@echo "  make install-system  - Instala dependências do sistema (espeak-ng)"
	@echo "  make install-deps    - Instala dependências Python com uv"
	@echo "  make run             - Inicia a aplicação"
	@echo "  make initialize      - Baixa os modelos pré-treinados necessários"
	@echo "  make clean           - Limpa arquivos temporários"

install-system:
	@echo "Instalando espeak-ng..."
	@if [ -x "$$(command -v apt-get)" ]; then \
		sudo apt-get update && sudo apt-get install -y espeak-ng; \
	elif [ -x "$$(command -v brew)" ]; then \
		brew install espeak; \
	else \
		echo "Gerenciador de pacotes não suportado. Por favor instale espeak-ng manualmente."; \
		exit 1; \
	fi
	@echo "espeak-ng instalado com sucesso!"

install-deps:
	@echo "Instalando dependências Python..."
	uv sync

run:
	@echo "Iniciando aplicação..."
	uv run python app.py --share --no_autolaunch

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

initialize:
	@echo "Baixando modelos pré-treinados..."
	uv run python initialize.py
