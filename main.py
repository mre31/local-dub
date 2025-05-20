import argparse
from pathlib import Path
import subprocess
from dotenv import load_dotenv
from audio_extractor import extract_audio_ffmpeg, separate_vocals_demucs_lib, get_audio_duration
from transcriber import transcribe_audio_whisper
from translator import translate_text_gemini
from voice_synthesizer import synthesize_text_coqui, adjust_audio_speed_ffmpeg
import json
import torch
import shutil

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Video dubbing tool.")
    parser.add_argument("video_path", type=str, help="Path to the video file to process.")
    parser.add_argument("--demucs_model", type=str, default="hdemucs_mmi", help="Demucs model to use for separation. Default: hdemucs_mmi (Hybrid Demucs v3)")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model for transcription. Default: medium")
    parser.add_argument("--transcribe_language", type=str, default=None, help="Language for Whisper transcription (e.g., en, tr). Default: auto-detect")
    parser.add_argument("--target_language", type=str, required=True, help="Target language CODE for translation and TTS (e.g., en, tr, de, es, fr). This is required. Use standard 2-letter codes.")
    parser.add_argument("--reference_voice_for_cloning", type=str, required=True, help="Path to the reference WAV file for voice cloning with Coqui TTS. This is required.")
    parser.add_argument("--tts_model", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="Coqui TTS model to use. Default: tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--cpu", action="store_true", help="Force processing on CPU, even if CUDA is available.")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before processing.")

    args = parser.parse_args()
    video_file = Path(args.video_path)
    demucs_model_name = args.demucs_model
    whisper_model_name = args.whisper_model
    transcribe_lang_whisper = args.transcribe_language
    target_language_code = args.target_language
    reference_voice_path_str = args.reference_voice_for_cloning
    coqui_tts_model_name = args.tts_model
    
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA selected, but no CUDA-enabled GPU is available. Falling back to CPU.")
        device = "cpu"
    elif device == "cuda":
        print("CUDA is available and will be used.")
    else:
        print(f"CPU will be used for Demucs, Whisper, and Coqui TTS if applicable based on their device handling.")

    if not video_file.is_file():
        print(f"Error: Video file not found or is not a file: {video_file}")
        return
    
    ref_voice_path = Path(reference_voice_path_str)
    if not ref_voice_path.is_file():
        print(f"Error: Reference voice file for cloning not found or is not a file: {ref_voice_path}")
        return

    output_directory = Path("output")

    if args.clean:
        if output_directory.exists():
            print(f"Cleaning output directory: {output_directory}")
            shutil.rmtree(output_directory)
    
    output_directory.mkdir(parents=True, exist_ok=True)
    
    raw_audio_filename = f"{video_file.stem}_audio.wav"
    raw_audio_path = output_directory / raw_audio_filename

    print(f"Processing video: {video_file}")
    print(f"Output directory: {output_directory}")
    print(f"Target language: {target_language_code}")
    print(f"Reference voice for cloning: {reference_voice_path_str}")
    print(f"Device for processing: {device.upper()}")


    try:
        print(f"Step 1: Extracting audio to: {raw_audio_path} using ffmpeg...")
        extract_audio_ffmpeg(str(video_file), str(raw_audio_path))
        print(f"Successfully extracted audio to: {raw_audio_path}")
    except Exception as e:
        print(f"Error during ffmpeg audio extraction: {e}")
        return

    vocals_output_path = output_directory / f"{video_file.stem}_vocals.wav"
    background_output_path = output_directory / f"{video_file.stem}_background.wav"

    try:
        print(f"Step 2: Separating vocals from {raw_audio_path} using Demucs model: {demucs_model_name} on {device}...")
        separate_vocals_demucs_lib(str(raw_audio_path), 
                                   str(vocals_output_path),
                                   str(background_output_path),
                                   model_name=demucs_model_name,
                                   device=device)
        print(f"Successfully separated audio using Demucs.")
        print(f"  Vocals (speech): {vocals_output_path}")
        print(f"  Background sound (no_vocals): {background_output_path}")
        if not vocals_output_path.exists():
             print(f"Warning: Vocals file not found at {vocals_output_path} after Demucs separation. Subsequent steps will be skipped.")
             return
        if not background_output_path.exists():
            print(f"Warning: Background (no_vocals) file not found at {background_output_path}. Speed adjustment for TTS might be affected or use raw audio duration.")


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
                                                      device=device,
                                                      language=transcribe_lang_whisper)
        
        if transcription_result and "text" in transcription_result and transcription_result["text"].strip():
            transcribed_text_content = transcription_result["text"]
            detected_language_whisper = transcription_result.get("language")
            print(f"Successfully transcribed audio using Whisper.")
            print(f"  Detected language by Whisper: {detected_language_whisper}")
            print(f"  Transcription (first 100 chars): \\n{transcribed_text_content[:100]}...")
            
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

    translated_text_content = None
    if transcribed_text_content:
        safe_target_lang_code_for_filename = target_language_code.lower().replace(' ', '_') 
        translation_output_filename = f"{video_file.stem}_translated_to_{safe_target_lang_code_for_filename}.txt"
        translation_output_path = output_directory / translation_output_filename
        source_lang_for_gemini = detected_language_whisper if detected_language_whisper else None 

        try:
            print(f"Step 4: Translating text to \'{target_language_code}\' using Gemini (Source lang for prompt: {source_lang_for_gemini or 'auto-detect'})...")
            translated_text_content = translate_text_gemini(transcribed_text_content, 
                                                target_language=target_language_code,
                                                source_language=source_lang_for_gemini)
            
            if translated_text_content:
                print(f"Successfully translated text using Gemini.")
                print(f"  Translated text (first 100 chars): \\n{translated_text_content[:100]}...")
                with open(translation_output_path, "w", encoding="utf-8") as f_trans:
                    f_trans.write(translated_text_content)
                print(f"  Full translated text saved to: {translation_output_path}")
            else:
                print(f"Gemini translation failed or returned no text. Skipping TTS.")
                return

        except Exception as e:
            print(f"An error occurred during Gemini translation: {e}")
            print("Please ensure GEMINI_API_KEY is set in .env and valid, and 'google-genai' is installed.")
            return
    else:
        print("Skipping translation and TTS steps as there was no transcribed text available.")
        return

    if translated_text_content:
        safe_target_lang_code_for_filename = target_language_code.lower()
        synthesized_audio_filename = f"{video_file.stem}_synthesized_to_{safe_target_lang_code_for_filename}.wav"
        synthesized_audio_path = output_directory / synthesized_audio_filename
        
        adjusted_audio_filename = f"{video_file.stem}_synthesized_to_{safe_target_lang_code_for_filename}_adjusted.wav"
        adjusted_audio_path = output_directory / adjusted_audio_filename

        try:
            print(f"Step 5: Synthesizing translated text to speech using Coqui TTS model: {coqui_tts_model_name} (Language: {target_language_code}) on {device}...")
            synthesis_successful = synthesize_text_coqui(
                text_to_synthesize=translated_text_content,
                output_wav_path_str=str(synthesized_audio_path),
                target_language=target_language_code,
                reference_wav_path_str=str(ref_voice_path),
                model_name=coqui_tts_model_name,
                device=device
            )
            if not synthesis_successful or not synthesized_audio_path.exists():
                print("Coqui TTS synthesis failed or output file not found. Skipping speed adjustment.")
                return
            print(f"Successfully synthesized audio to: {synthesized_audio_path}")

            print(f"Step 6: Adjusting speed of synthesized audio ({synthesized_audio_path}) to match duration of background audio ({background_output_path})...")
            
            target_duration_audio_path = background_output_path
            if not target_duration_audio_path.exists():
                print(f"Warning: Target duration audio file ({target_duration_audio_path}) not found. Trying raw extracted audio ({raw_audio_path}) for duration instead.")
                target_duration_audio_path = raw_audio_path
                if not target_duration_audio_path.exists():
                    print(f"Error: Neither background audio nor raw audio found for determining target duration. Skipping speed adjustment.")
                    return


            target_duration = get_audio_duration(str(target_duration_audio_path))

            if target_duration is None:
                print(f"Could not get duration for {target_duration_audio_path}. Skipping speed adjustment.")
                return
            
            print(f"Target duration for synthesized audio (from {target_duration_audio_path.name}): {target_duration:.2f} seconds.")

            adjustment_successful = adjust_audio_speed_ffmpeg(
                input_audio_path_str=str(synthesized_audio_path),
                output_audio_path_str=str(adjusted_audio_path),
                target_duration_seconds=target_duration
            )
            if not adjustment_successful:
                print("Failed to adjust audio speed.")
            else:
                print(f"Successfully adjusted synthesized audio speed. Final dubbed audio track (voice only): {adjusted_audio_path}")
                print("Next step would be to merge this with the background audio.")

        except Exception as e:
            print(f"An error occurred during Coqui TTS synthesis or speed adjustment: {e}")
            return
    else:
        print("Skipping TTS step as there was no translated text available.")


    print("All processing steps complete.")

if __name__ == "__main__":
    main()