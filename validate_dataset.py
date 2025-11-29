#!/usr/bin/env python3
"""
Script para validar e corrigir nomes de arquivo inválidos no dataset do Style-Bert-VITS2.

Este script:
1. Verifica arquivos de áudio com nomes problemáticos (começando com -, contendo caracteres especiais)
2. Renomeia automaticamente arquivos problemáticos
3. Atualiza referências nos arquivos .list (esd.list, train.list, val.list)
4. Gera relatório de alterações

Uso:
    python validate_dataset.py --model_name arteDaGuerra [--dry-run] [--dataset-root /path/to/Data]
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from style_bert_vits2.logging import logger


def is_invalid_filename(filename: str) -> bool:
    """
    Verifica se um nome de arquivo é inválido.
    
    Nomes inválidos incluem:
    - Começando com hífen (-)
    - Começando com ponto (.)
    - Contendo caracteres especiais problemáticos
    """
    # Verifica se começa com hífen ou ponto
    if filename.startswith('-') or filename.startswith('.'):
        return True
    
    # Verifica caracteres problemáticos (exceto extensão)
    name_without_ext = Path(filename).stem
    # Permite apenas letras, números, underscore e hífen no meio do nome
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]*$', name_without_ext):
        return True
    
    return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza um nome de arquivo removendo/substituindo caracteres problemáticos.
    
    Args:
        filename: Nome do arquivo original
        
    Returns:
        Nome do arquivo sanitizado
    """
    path = Path(filename)
    name = path.stem
    ext = path.suffix
    
    # Remove hífens e pontos do início
    name = name.lstrip('-.')
    
    # Se o nome ficou vazio, usa um prefixo padrão
    if not name:
        name = "audio_file"
    
    # Substitui caracteres especiais por underscore
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    
    # Remove underscores múltiplos
    name = re.sub(r'_+', '_', name)
    
    # Remove underscores do início e fim
    name = name.strip('_')
    
    return f"{name}{ext}"


def find_invalid_files(raw_dir: Path) -> List[Tuple[Path, str]]:
    """
    Encontra todos os arquivos com nomes inválidos no diretório raw.
    
    Args:
        raw_dir: Diretório raw do dataset
        
    Returns:
        Lista de tuplas (caminho_completo, caminho_relativo)
    """
    invalid_files = []
    
    # Procura por arquivos de áudio
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    for ext in audio_extensions:
        for audio_file in raw_dir.rglob(f'*{ext}'):
            # Obtém caminho relativo ao raw_dir
            rel_path = audio_file.relative_to(raw_dir)
            
            # Verifica se alguma parte do caminho tem nome inválido
            if any(is_invalid_filename(part) for part in rel_path.parts):
                invalid_files.append((audio_file, str(rel_path)))
    
    return invalid_files


def rename_files(
    invalid_files: List[Tuple[Path, str]], 
    raw_dir: Path,
    dry_run: bool = False
) -> Dict[str, str]:
    """
    Renomeia arquivos inválidos.
    
    Args:
        invalid_files: Lista de arquivos inválidos
        raw_dir: Diretório raw do dataset
        dry_run: Se True, apenas simula as mudanças
        
    Returns:
        Dicionário mapeando caminho antigo -> caminho novo (relativo ao raw_dir)
    """
    rename_map = {}
    
    for full_path, rel_path in invalid_files:
        # Sanitiza cada parte do caminho
        parts = Path(rel_path).parts
        new_parts = [sanitize_filename(part) if part != parts[-1] else part 
                     for part in parts[:-1]]
        new_parts.append(sanitize_filename(parts[-1]))
        
        new_rel_path = Path(*new_parts)
        new_full_path = raw_dir / new_rel_path
        
        # Garante que o diretório de destino existe
        if not dry_run:
            new_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte extensão para .wav no mapeamento (como esperado pelo esd.list)
        old_list_path = str(Path(rel_path).with_suffix('.wav'))
        new_list_path = str(new_rel_path.with_suffix('.wav'))
        
        rename_map[old_list_path] = new_list_path
        
        if dry_run:
            logger.info(f"[DRY RUN] Renomearia: {rel_path} -> {new_rel_path}")
        else:
            logger.info(f"Renomeando: {rel_path} -> {new_rel_path}")
            shutil.move(str(full_path), str(new_full_path))
            
            # Renomeia também arquivos .bert.pt associados, se existirem
            bert_file = full_path.with_suffix('.bert.pt')
            if bert_file.exists():
                new_bert_file = new_full_path.with_suffix('.bert.pt')
                shutil.move(str(bert_file), str(new_bert_file))
                logger.info(f"  Renomeado também: {bert_file.name} -> {new_bert_file.name}")
    
    return rename_map


def update_list_file(
    list_file: Path, 
    rename_map: Dict[str, str],
    dry_run: bool = False
) -> int:
    """
    Atualiza um arquivo .list com os novos nomes de arquivo.
    
    Args:
        list_file: Caminho para o arquivo .list
        rename_map: Dicionário de mapeamento antigo -> novo
        dry_run: Se True, apenas simula as mudanças
        
    Returns:
        Número de linhas atualizadas
    """
    if not list_file.exists():
        logger.warning(f"Arquivo {list_file} não encontrado, pulando...")
        return 0
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    updates_count = 0
    
    for line in lines:
        if not line.strip():
            updated_lines.append(line)
            continue
            
        parts = line.strip().split('|')
        if len(parts) >= 1:
            old_path = parts[0]
            
            if old_path in rename_map:
                parts[0] = rename_map[old_path]
                updates_count += 1
                if dry_run:
                    logger.info(f"[DRY RUN] Atualizaria em {list_file.name}: {old_path} -> {parts[0]}")
                else:
                    logger.info(f"Atualizado em {list_file.name}: {old_path} -> {parts[0]}")
        
        updated_lines.append('|'.join(parts) + '\n')
    
    if not dry_run and updates_count > 0:
        # Faz backup do arquivo original
        backup_file = list_file.with_suffix(list_file.suffix + '.backup')
        shutil.copy(str(list_file), str(backup_file))
        logger.info(f"Backup criado: {backup_file}")
        
        # Salva arquivo atualizado
        with open(list_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
    
    return updates_count


def main():
    parser = argparse.ArgumentParser(
        description='Valida e corrige nomes de arquivo inválidos no dataset'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Nome do modelo (pasta em Data/)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='./Data',
        help='Diretório raiz do dataset (padrão: ./Data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apenas simula as mudanças sem aplicá-las'
    )
    
    args = parser.parse_args()
    
    # Configura caminhos
    dataset_root = Path(args.dataset_root)
    model_dir = dataset_root / args.model_name
    raw_dir = model_dir / 'raw'
    
    if not raw_dir.exists():
        logger.error(f"Diretório {raw_dir} não encontrado!")
        return 1
    
    logger.info(f"Validando dataset em: {raw_dir}")
    
    # Encontra arquivos inválidos
    invalid_files = find_invalid_files(raw_dir)
    
    if not invalid_files:
        logger.success("✓ Nenhum arquivo com nome inválido encontrado!")
        return 0
    
    logger.warning(f"Encontrados {len(invalid_files)} arquivo(s) com nome inválido")
    
    # Renomeia arquivos
    rename_map = rename_files(invalid_files, raw_dir, dry_run=args.dry_run)
    
    # Atualiza arquivos .list
    list_files = [
        model_dir / 'esd.list',
        model_dir / 'train.list',
        model_dir / 'val.list'
    ]
    
    total_updates = 0
    for list_file in list_files:
        updates = update_list_file(list_file, rename_map, dry_run=args.dry_run)
        total_updates += updates
    
    # Relatório final
    logger.info("\n" + "="*60)
    if args.dry_run:
        logger.info("SIMULAÇÃO CONCLUÍDA (nenhuma mudança foi aplicada)")
    else:
        logger.success("VALIDAÇÃO CONCLUÍDA")
    logger.info(f"Arquivos renomeados: {len(rename_map)}")
    logger.info(f"Linhas atualizadas em .list: {total_updates}")
    logger.info("="*60)
    
    if args.dry_run:
        logger.info("\nExecute sem --dry-run para aplicar as mudanças")
    
    return 0


if __name__ == '__main__':
    exit(main())
