import subprocess
from pathlib import Path
import torch
from demucs.api import Separator, save_audio
import json # For ffprobe output parsing

def extract_audio_ffmpeg(video_path_str: str, output_audio_path_str: str):
    video_path = Path(video_path_str)
    output_audio_path = Path(output_audio_path_str)

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-y",
        str(output_audio_path)
    ]
    _ = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')

def separate_vocals_demucs_lib(input_audio_path_str: str, 
                               vocals_output_path_str: str, 
                               background_output_path_str: str, 
                               model_name: str = "mdx_extra_q",
                               device: str = "cuda"):
    input_audio_path = Path(input_audio_path_str)
    vocals_output_path = Path(vocals_output_path_str)
    background_output_path = Path(background_output_path_str)

    separator = Separator(model=model_name, device=device)
    
    vocals_output_path.parent.mkdir(parents=True, exist_ok=True)
    background_output_path.parent.mkdir(parents=True, exist_ok=True)

    _, separated_stems = separator.separate_audio_file(str(input_audio_path))

    if 'vocals' not in separated_stems:
        raise RuntimeError(f"Vocals stem not found in Demucs model '{model_name}' output. Available stems: {list(separated_stems.keys())}")

    vocals_tensor = separated_stems['vocals']
    save_audio(vocals_tensor, str(vocals_output_path), samplerate=separator.samplerate)

    background_tensor = torch.zeros_like(vocals_tensor)
    
    for stem_name, stem_tensor in separated_stems.items():
        if stem_name != 'vocals':
            background_tensor += stem_tensor
            
    save_audio(background_tensor, str(background_output_path), samplerate=separator.samplerate)

def get_audio_duration(audio_path_str: str) -> float | None:
    """Gets the duration of an audio file using ffprobe."""
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"Error: Audio file not found for duration check: {audio_path}")
        return None
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration with ffprobe for {audio_path}: {e}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: ffprobe command not found. Please ensure ffmpeg (which includes ffprobe) is installed and in PATH.")
        return None
    except ValueError as e:
        print(f"Error parsing duration from ffprobe output for {audio_path}: {e}\nStdout: {result.stdout}")
        return None