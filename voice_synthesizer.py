from pathlib import Path
import subprocess
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import torch.serialization

DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def synthesize_text_coqui(text_to_synthesize: str, 
                            output_wav_path_str: str, 
                            target_language: str, 
                            reference_wav_path_str: str = None,
                            model_name: str = DEFAULT_TTS_MODEL,
                            device: str = "cuda"):
    
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        safe_globals_to_add = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
        torch.serialization.add_safe_globals(safe_globals_to_add)
        print(f"TTS Info: Added {', '.join(c.__name__ for c in safe_globals_to_add)} to torch safe globals for compatibility.")
    else:
        print("TTS Warning: torch.serialization.add_safe_globals not found. This might be an older PyTorch version or an unexpected setup. TTS loading might fail on newer PyTorch if this is missing.")

    output_wav_path = Path(output_wav_path_str)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    actual_device = device
    if actual_device == "cuda" and not torch.cuda.is_available():
        print("TTS Warning: CUDA selected, but no CUDA-enabled GPU is available. Falling back to CPU for TTS.")
        actual_device = "cpu"
    elif actual_device == "cuda":
        print("TTS Info: CUDA is available and will be used for Coqui TTS.")
    else:
        print("TTS Info: CPU will be used for Coqui TTS.")

    try:
        
        tts = TTS(model_name=model_name, progress_bar=True)
        tts.to(actual_device)
        print(f"TTS Info: Model {model_name} loaded on {actual_device}.")
        
        supported_xtts_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
        if target_language.lower() not in supported_xtts_languages:
            print(f"TTS Error: Language code '{target_language}' is not in the supported list for XTTS v2: {supported_xtts_languages}")
            print("Please use one of the supported language codes (e.g., 'tr' for Turkish, 'en' for English).")
            return False

        effective_target_language = target_language.lower()

        if reference_wav_path_str and Path(reference_wav_path_str).exists():
            print(f"Synthesizing with voice cloning from: {reference_wav_path_str} for language: {effective_target_language}")
            tts.tts_to_file(
                text=text_to_synthesize, 
                speaker_wav=str(reference_wav_path_str), 
                language=effective_target_language, 
                file_path=str(output_wav_path)
            )
        else:
            if reference_wav_path_str:
                print(f"Warning: Reference voice file not found at {reference_wav_path_str}. Using default voice.")
            print(f"Synthesizing with default voice for language: {effective_target_language}")
            tts.tts_to_file(
                text=text_to_synthesize, 
                language=effective_target_language, 
                file_path=str(output_wav_path)
            )
        print(f"Successfully synthesized audio to: {output_wav_path}")
        return True
    except Exception as e:
        print(f"Error during Coqui TTS synthesis: {e}")
        return False

def adjust_audio_speed_ffmpeg(input_audio_path_str: str, 
                                output_audio_path_str: str, 
                                target_duration_seconds: float):
    
    input_audio_path = Path(input_audio_path_str)
    output_audio_path = Path(output_audio_path_str)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_audio_path.exists():
        print(f"Error: Input audio for speed adjustment not found: {input_audio_path}")
        return False
    
    try:
        
        probe_command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(input_audio_path)
        ]
        result = subprocess.run(probe_command, check=True, capture_output=True, text=True, encoding='utf-8')
        current_duration = float(result.stdout.strip())

        if current_duration <= 0:
            print(f"Error: Could not determine current duration or duration is zero for {input_audio_path}")
            return False
        if target_duration_seconds <= 0:
            print(f"Error: Target duration ({target_duration_seconds}s) is invalid.")
            return False

        speed_factor = current_duration / target_duration_seconds
        if not (0.5 <= speed_factor <= 100.0):
            print(f"Warning: Required speed factor {speed_factor:.2f} is outside ffmpeg's direct atempo range [0.5, 100.0]. Capping speed factor.")
            print("         The final audio duration might not exactly match the target duration.")
            speed_factor = max(0.5, min(100.0, speed_factor))

        ffmpeg_command = [
            "ffmpeg", "-i", str(input_audio_path),
            "-filter:a", f"atempo={speed_factor:.4f}", 
            "-vn", "-y", str(output_audio_path)
        ]
        
        print(f"Adjusting audio speed. Current: {current_duration:.2f}s, Target: {target_duration_seconds:.2f}s, Factor: {speed_factor:.4f}")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Successfully adjusted audio speed and saved to: {output_audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg speed adjustment: {e}\nStderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg/ffprobe command not found for speed adjustment.")
        return False
    except ValueError as e:
        print(f"Error parsing duration for speed adjustment: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during speed adjustment: {e}")
        return False

if __name__ == '__main__':
    
    
    print("Testing Voice Synthesizer...")
    test_text = "Hello, this is a test voiceover made with Coqui TTS."
    target_lang_code = "en"
    temp_synth_path = Path("output/temp_synthesized_audio.wav")
    final_adjusted_path = Path("output/final_adjusted_audio.wav")
    example_reference_voice = None

    if synthesize_text_coqui(test_text, str(temp_synth_path), target_lang_code, 
                             reference_wav_path_str=str(example_reference_voice) if example_reference_voice else None,
                             device="cpu"):
        print("Initial synthesis successful.")
        target_dur = 5.0
        if adjust_audio_speed_ffmpeg(str(temp_synth_path), str(final_adjusted_path), target_dur):
            print(f"Speed adjustment successful. Final audio at: {final_adjusted_path}")
        else:
            print("Speed adjustment failed.")
    else:
        print("Initial synthesis failed.") 