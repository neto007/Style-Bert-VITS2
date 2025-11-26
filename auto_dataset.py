import argparse
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Automate dataset creation for Style-Bert-VITS2 (Portuguese)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model/speaker")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw audio files")
    parser.add_argument("--initial_prompt", type=str, default="Olá! Tudo bem com você?", help="Initial prompt for Whisper")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    input_dir = Path(args.input_dir).resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "Data" / model_name
    raw_dir = data_dir / "raw"
    
    print(f"Preparing dataset for model: {model_name}")
    print(f"Input audio: {input_dir}")
    print(f"Target directory: {data_dir}")
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy audio files to raw directory
    print("Copying audio files...")
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    count = 0
    for ext in audio_extensions:
        for file in input_dir.glob(f"*{ext}"):
            dest = raw_dir / file.name
            if not dest.exists():
                # Use standard copy (shutil would be better but trying to keep imports minimal if possible, actually shutil is standard)
                import shutil
                shutil.copy2(file, dest)
                count += 1
    
    print(f"Copied {count} audio files to {raw_dir}")
    
    if count == 0:
        # Check if files are already there
        existing = list(raw_dir.glob("*"))
        if len(existing) > 0:
            print(f"Found {len(existing)} files already in {raw_dir}. Proceeding...")
        else:
            print("No audio files found to process. Exiting.")
            return

    # Run transcription
    print("Running transcription...")
    cmd = [
        "python", "transcribe.py",
        "--model_name", model_name,
        "--language", "pt",
        "--initial_prompt", args.initial_prompt
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Transcription complete!")
        print(f"Dataset created at: {data_dir}")
        print(f"Transcription file: {data_dir}/esd.list")
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()
