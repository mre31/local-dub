import argparse
from pathlib import Path
import subprocess
from dotenv import load_dotenv
from audio_extractor import extract_audio_ffmpeg, separate_vocals_demucs_lib
from transcriber import transcribe_audio_whisper
from translator import translate_text_gemini
import json
import torch

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Video dubbing tool.")
    parser.add_argument("video_path", type=str, help="Path to the video file to process.")
    parser.add_argument("--demucs_model", type=str, default="hdemucs_mmi", help="Demucs model to use for separation. Default: hdemucs_mmi (Hybrid Demucs v3)")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model for transcription. Default: medium")
    parser.add_argument("--transcribe_language", type=str, default=None, help="Language for Whisper transcription (e.g., en, tr). Default: auto-detect")
    parser.add_argument("--target_language", type=str, required=True, help="Target language for translation (e.g., English, Turkish, German). This is required.")
    parser.add_argument("--cpu", action="store_true", help="Force processing on CPU, even if CUDA is available.")

    args = parser.parse_args()
    video_file = Path(args.video_path)
    demucs_model_name = args.demucs_model
    whisper_model_name = args.whisper_model
    transcribe_lang_whisper = args.transcribe_language
    target_translation_language = args.target_language
    
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA selected, but no CUDA-enabled GPU is available. Falling back to CPU.")
        device = "cpu"
    elif device == "cuda":
        print("CUDA is available and will be used.")
    else:
        print("CPU will be used for Demucs and Whisper if applicable.")

    if not video_file.is_file():
        print(f"Error: Video file not found or is not a file: {video_file}")
        return

    output_directory = Path("output")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    raw_audio_filename = f"{video_file.stem}_audio.wav"
    raw_audio_path = output_directory / raw_audio_filename

    print(f"Processing video: {video_file}")
    print(f"Output directory: {output_directory}")

    try:
        print(f"Step 1: Extracting audio to: {raw_audio_path} using ffmpeg...")
        extract_audio_ffmpeg(str(video_file), str(raw_audio_path))
        print(f"Successfully extracted audio to: {raw_audio_path}")
    except Exception as e:
        print(f"Error during ffmpeg audio extraction: {e}")
        return

    vocals_output_path = output_directory / f"{raw_audio_path.stem}_vocals.wav"
    background_output_path = output_directory / f"{raw_audio_path.stem}_background.wav"

    try:
        print(f"Step 2: Separating vocals from {raw_audio_path} using Demucs model: {demucs_model_name} on {device}...")
        separate_vocals_demucs_lib(str(raw_audio_path), 
                                   str(vocals_output_path),
                                   str(background_output_path),
                                   model_name=demucs_model_name,
                                   device=device)
        print(f"Successfully separated audio using Demucs.")
        print(f"  Vocals (speech): {vocals_output_path}")
        print(f"  Background sound: {background_output_path}")
        if not vocals_output_path.exists():
             print(f"Warning: Vocals file not found at {vocals_output_path} after Demucs separation. Transcription & Translation steps will be skipped.")
             return
    except Exception as e:
        print(f"Error during Demucs vocal separation: {e}")
        return

    transcription_output_json_path = output_directory / f"{video_file.stem}_transcription.json"
    transcribed_text_content = None
    detected_language_whisper = None

    try:
        print(f"Step 3: Transcribing vocals from {vocals_output_path} using Whisper model: {whisper_model_name} (Language: {transcribe_lang_whisper or 'auto-detect'}) on {device}...")
        transcription_result = transcribe_audio_whisper(str(vocals_output_path), 
                                                      model_name=whisper_model_name, 
                                                      device=device)
        
        if transcription_result and "text" in transcription_result:
            transcribed_text_content = transcription_result["text"]
            detected_language_whisper = transcription_result.get("language", "N/A")
            print(f"Successfully transcribed audio using Whisper.")
            print(f"  Detected language by Whisper: {detected_language_whisper}")
            print(f"  Transcription (first 100 chars): \n{transcribed_text_content[:100]}...")
            
            with open(transcription_output_json_path, "w", encoding="utf-8") as f_json:
                json.dump(transcription_result, f_json, ensure_ascii=False, indent=4)
            print(f"  Full transcription result saved to: {transcription_output_json_path}")
        else:
            print(f"Whisper transcription failed or did not return expected text output for {vocals_output_path}.")
            if transcription_result is not None:
                print(f"  Whisper raw result: {transcription_result}")
            return
    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        return

    if transcribed_text_content:
        translation_output_path = output_directory / f"{video_file.stem}_translated_to_{target_translation_language.lower().replace(' ', '_')}.txt"
        source_lang_for_gemini = detected_language_whisper if detected_language_whisper and detected_language_whisper != "N/A" else None

        try:
            print(f"Step 4: Translating text to '{target_translation_language}' using Gemini (Source lang for prompt: {source_lang_for_gemini or 'auto-detect'})...")
            translated_text = translate_text_gemini(transcribed_text_content, 
                                                target_language=target_translation_language,
                                                source_language=source_lang_for_gemini)
            
            if translated_text:
                print(f"Successfully translated text using Gemini.")
                print(f"  Translated text (first 100 chars): \n{translated_text[:100]}...")
                with open(translation_output_path, "w", encoding="utf-8") as f_trans:
                    f_trans.write(translated_text)
                print(f"  Full translated text saved to: {translation_output_path}")
            else:
                print(f"Gemini translation failed or returned no text.")

        except Exception as e:
            print(f"An error occurred during Gemini translation: {e}")
            print("Please ensure GEMINI_API_KEY is set in .env and valid, and 'google-genai' is installed.")
    else:
        print("Skipping translation step as there was no transcribed text available.")

    print("All processing steps complete.")

if __name__ == "__main__":
    main() 