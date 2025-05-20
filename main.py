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

# Helper function to merge transcription segments
def merge_transcription_segments(original_segments, min_duration_seconds=30):
    if not original_segments:
        return []

    merged_segments_list = []
    current_processing_idx = 0
    new_segment_id_counter = 0

    while current_processing_idx < len(original_segments):
        group_start_segment = original_segments[current_processing_idx]
        
        merged_text_parts = [group_start_segment.get("text", "").strip()]
        merged_tokens = list(group_start_segment.get("tokens", []))
        
        current_start_time = group_start_segment.get("start", 0.0)
        current_end_time = group_start_segment.get("end", 0.0)
        
        # Accumulate subsequent segments for this group
        next_segment_in_group_idx = current_processing_idx + 1
        while next_segment_in_group_idx < len(original_segments):
            duration_so_far = current_end_time - current_start_time
            if duration_so_far >= min_duration_seconds:
                break 

            segment_to_add = original_segments[next_segment_in_group_idx]
            merged_text_parts.append(segment_to_add.get("text", "").strip())
            if "tokens" in segment_to_add and segment_to_add["tokens"] is not None:
                merged_tokens.extend(segment_to_add["tokens"])
            current_end_time = segment_to_add.get("end", current_end_time) 
            
            next_segment_in_group_idx += 1
        
        final_text = " ".join(filter(None, merged_text_parts)) # filter(None, ...) to remove empty strings before joining
        new_merged_segment = {
            "id": new_segment_id_counter,
            "seek": group_start_segment.get("seek", 0),
            "start": current_start_time,
            "end": current_end_time,
            "text": final_text,
            "tokens": merged_tokens,
            "temperature": group_start_segment.get("temperature", 0.0),
            "avg_logprob": group_start_segment.get("avg_logprob", 0.0),
            "compression_ratio": group_start_segment.get("compression_ratio", 0.0),
            "no_speech_prob": group_start_segment.get("no_speech_prob", 0.0)
        }
        merged_segments_list.append(new_merged_segment)
        new_segment_id_counter += 1
        
        current_processing_idx = next_segment_in_group_idx 

    return merged_segments_list

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

            # Save the original transcription result FIRST
            original_transcription_output_json_path = output_directory / f"{video_file.stem}_transcription_original_segments.json"
            try:
                with open(original_transcription_output_json_path, "w", encoding="utf-8") as f_json_orig:
                    json.dump(transcription_result, f_json_orig, ensure_ascii=False, indent=4)
                print(f"  Original transcription result saved to: {original_transcription_output_json_path}")
            except Exception as e_save_orig:
                print(f"  Warning: Could not save original transcription result: {e_save_orig}")
            
            # Merge segments before saving the primary transcription file
            original_segments = transcription_result.get("segments", [])
            if original_segments:
                print("Whisper Info: Merging transcription segments to a minimum of 30 seconds each...")
                merged_segments = merge_transcription_segments(original_segments, min_duration_seconds=30)
                transcription_result["segments"] = merged_segments
                print(f"Whisper Info: Segments merged. New segment count: {len(merged_segments)}")
            else:
                print("Whisper Warning: No segments found in transcription result to merge.")

            print(f"Successfully transcribed audio using Whisper.")
            print(f"  Detected language by Whisper: {detected_language_whisper}")
            print(f"  Transcription (first 100 chars): \\n{transcribed_text_content[:100]}...")
            
            with open(transcription_output_json_path, "w", encoding="utf-8") as f_json:
                json.dump(transcription_result, f_json, ensure_ascii=False, indent=4)
            print(f"  Full transcription result (with merged segments) saved to: {transcription_output_json_path}")
        else:
            print(f"Whisper transcription failed or did not return expected text output for {vocals_output_path}.")
            if transcription_result is not None:
                print(f"  Whisper raw result: {transcription_result}")
            return
    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        return

    translated_segments_dict = None
    segments_for_translation = None
    if transcribed_text_content:
        # Prepare segments for translation. Use the merged segments.
        segments_for_translation = [
            {
                "id": seg.get("id"), 
                "text": seg.get("text", "").strip(),
                "start": seg.get("start"), # Ensure start time is included
                "end": seg.get("end")      # Ensure end time is included
            }
            for seg in transcription_result["segments"] 
            # Ensure segments have text and valid start/end times for duration calculation
            if seg.get("text", "").strip() and seg.get("start") is not None and seg.get("end") is not None
        ]
        
        if not segments_for_translation:
            print("No valid segments with text found in transcription to translate. Skipping translation and TTS.")
            return # Exit if no segments to translate

        safe_target_lang_code_for_filename = target_language_code.lower().replace(' ', '_') 
        # New filename for the JSON output of translated segments
        translation_output_json_path = output_directory / f"{video_file.stem}_translated_segments_to_{safe_target_lang_code_for_filename}.json"
        # Old filename for the plain text full translation (will still be generated for TTS)
        translation_output_txt_path = output_directory / f"{video_file.stem}_translated_full_to_{safe_target_lang_code_for_filename}.txt"
        
        source_lang_for_gemini = detected_language_whisper if detected_language_whisper else None 

        try:
            print(f"Step 4: Translating segments to '{target_language_code}' using Gemini (Source lang for prompt: {source_lang_for_gemini or 'auto-detect'})...")
            # Call the updated translator function
            translated_segments_dict = translate_text_gemini(
                segments_to_translate=segments_for_translation, 
                target_language=target_language_code,
                source_language=source_lang_for_gemini
            )
            
            if translated_segments_dict:
                print(f"Successfully received translated segments from Gemini.")
                # Save the structured translated segments to a JSON file
                with open(translation_output_json_path, "w", encoding="utf-8") as f_trans_json:
                    json.dump(translated_segments_dict, f_trans_json, ensure_ascii=False, indent=4)
                print(f"  Translated segments saved to JSON: {translation_output_json_path}")

                # Reconstruct the full translated text for TTS and plain text saving
                # Ensure original segment order is preserved if possible, map by ID
                translated_text_parts = []
                for original_segment in segments_for_translation: # Iterate in original order
                    segment_id_str = str(original_segment["id"])
                    if segment_id_str in translated_segments_dict:
                        translated_text_parts.append(translated_segments_dict[segment_id_str])
                    else:
                        print(f"Warning: Translated text for segment ID {segment_id_str} not found in Gemini's response.")
                
                translated_text_content = " ".join(filter(None, translated_text_parts)).strip()

                if translated_text_content:
                    print(f"  Reconstructed translated text (first 100 chars): \\n{translated_text_content[:100]}...")
                    with open(translation_output_txt_path, "w", encoding="utf-8") as f_trans_txt:
                        f_trans_txt.write(translated_text_content)
                    print(f"  Full reconstructed translated text saved to: {translation_output_txt_path}")
                else:
                    print("Warning: Reconstructed translated text is empty. TTS might be skipped or fail.")
                    # We might want to return here if no text content for TTS
            else:
                print(f"Gemini translation returned no data or failed. Skipping TTS.")
                return

        except Exception as e:
            print(f"An error occurred during Gemini translation: {e}")
            print("Please ensure GEMINI_API_KEY is set in .env and valid, and 'google-genai' is installed.")
            return
    else:
        print("Skipping translation and TTS steps as there was no transcribed text available.")
        return

    # Initialize these before the main TTS/Merging logic block
    final_dubbed_voice_track_path = None
    final_video_assembly_possible = False

    # STEP 5 & 6: Segment-wise TTS, Speed Adjustment, and Final Audio Merging
    if translated_segments_dict and segments_for_translation: # Use translated_segments_dict and segments_for_translation
        print(f"Step 5 & 6: Segment-wise TTS, Speed Adjustment, and Final Audio Merging...")
        safe_target_lang_code_for_filename = target_language_code.lower()
        
        temp_segments_dir = output_directory / "temp_audio_segments"
        temp_segments_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Temporary directory for audio segments: {temp_segments_dir}")

        adjusted_segment_files = [] 

        for i, original_segment_info in enumerate(segments_for_translation):
            segment_id = original_segment_info["id"]
            segment_id_str = str(segment_id)
            original_text = original_segment_info["text"]
            original_start_time = original_segment_info["start"]
            original_end_time = original_segment_info["end"]

            if segment_id_str not in translated_segments_dict or not translated_segments_dict[segment_id_str].strip():
                print(f"  Skipping TTS for segment ID {segment_id_str}: No translated text found or text is empty.")
                continue
            
            translated_segment_text = translated_segments_dict[segment_id_str]
            target_segment_duration = original_end_time - original_start_time
            if target_segment_duration <= 0:
                print(f"  Warning: Segment ID {segment_id_str} has a non-positive target duration ({target_segment_duration:.2f}s). Skipping this segment for TTS.")
                continue

            print(f"  Processing segment {i+1}/{len(segments_for_translation)} (ID: {segment_id_str}): Target duration {target_segment_duration:.2f}s")
            
            # Define paths for this segment's audio files
            segment_raw_synth_filename = f"{video_file.stem}_segment_{segment_id_str}_raw.wav"
            segment_raw_synth_path = temp_segments_dir / segment_raw_synth_filename
            
            # New path for silence-removed audio
            segment_silence_removed_filename = f"{video_file.stem}_segment_{segment_id_str}_silenceremoved.wav"
            segment_silence_removed_path = temp_segments_dir / segment_silence_removed_filename
            
            segment_adjusted_filename = f"{video_file.stem}_segment_{segment_id_str}_adjusted.wav"
            segment_adjusted_path = temp_segments_dir / segment_adjusted_filename

            try:
                print(f"    Synthesizing segment ID {segment_id_str} to '{target_language_code}'...")
                synthesis_successful = synthesize_text_coqui(
                    text_to_synthesize=translated_segment_text,
                    output_wav_path_str=str(segment_raw_synth_path),
                    target_language=target_language_code,
                    reference_wav_path_str=str(ref_voice_path),
                    model_name=coqui_tts_model_name,
                    device=device
                )
                if not synthesis_successful or not segment_raw_synth_path.exists():
                    print(f"    Coqui TTS synthesis failed or output file not found for segment ID {segment_id_str}. Skipping further processing for this segment.")
                    continue 
                print(f"    Successfully synthesized segment ID {segment_id_str} to: {segment_raw_synth_path}")

                # Step: Remove silence from the synthesized segment
                print(f"    Removing silence from segment ID {segment_id_str} ({segment_raw_synth_path})...")
                silence_removal_command = [
                    "ffmpeg",
                    "-i", str(segment_raw_synth_path),
                    "-af", "silenceremove=start_periods=1:start_threshold=-35dB:stop_periods=1:stop_threshold=-35dB:stop_duration=0.3:leave_silence=0.05:detection=peak",
                    "-y", str(segment_silence_removed_path)
                ]
                try:
                    subprocess.run(silence_removal_command, check=True, capture_output=True, text=True, encoding='utf-8')
                    if not segment_silence_removed_path.exists() or segment_silence_removed_path.stat().st_size == 0:
                        print(f"    Warning: Silence removal for segment ID {segment_id_str} resulted in an empty or non-existent file. Using raw synthesized audio for speed adjustment.")
                        # Fallback: use the raw synthesized path if silence removal fails to produce a valid file
                        # This means segment_silence_removed_path might not be the one used by adjust_audio_speed_ffmpeg in this case
                        # We will pass segment_raw_synth_path to adjust_audio_speed_ffmpeg if silence removal fails to produce a file
                        # However, for simplicity, we'll assume if it runs without error, output is fine, but check existence.
                        # A better fallback is to ensure the input to speed adjustment is always valid.
                        # For now, if silence removal seems to fail (e.g. empty file), we can log it.
                        # Let's make sure the input for speed adjustment is the silence-removed one if it exists and is valid.
                        input_for_speed_adjustment = segment_silence_removed_path if segment_silence_removed_path.exists() and segment_silence_removed_path.stat().st_size > 0 else segment_raw_synth_path
                        if input_for_speed_adjustment != segment_silence_removed_path:
                             print(f"    Using {input_for_speed_adjustment.name} for speed adjustment due to silence removal issue.")
                    else:
                        print(f"    Successfully removed silence for segment ID {segment_id_str}. Output: {segment_silence_removed_path}")
                        input_for_speed_adjustment = segment_silence_removed_path

                except subprocess.CalledProcessError as e_silence:
                    print(f"    Error during ffmpeg silence removal for segment ID {segment_id_str}: {e_silence}\n    Stderr: {e_silence.stderr}")
                    print(f"    Using raw synthesized audio ({segment_raw_synth_path.name}) for speed adjustment due to silence removal error.")
                    input_for_speed_adjustment = segment_raw_synth_path # Fallback to raw if error
                except FileNotFoundError:
                    print("    Error: ffmpeg command not found for silence removal.")
                    print(f"    Using raw synthesized audio ({segment_raw_synth_path.name}) for speed adjustment.")
                    input_for_speed_adjustment = segment_raw_synth_path # Fallback to raw if ffmpeg not found

                print(f"    Adjusting speed of segment ID {segment_id_str} (input: {input_for_speed_adjustment.name}) to {target_segment_duration:.2f}s...")
                adjustment_successful = adjust_audio_speed_ffmpeg(
                    input_audio_path_str=str(input_for_speed_adjustment), # Use silence-removed (or fallback) audio
                    output_audio_path_str=str(segment_adjusted_path),
                    target_duration_seconds=target_segment_duration
                )
                if not adjustment_successful or not segment_adjusted_path.exists():
                    print(f"    Failed to adjust audio speed for segment ID {segment_id_str}.")
                    continue # Skip to next segment
                
                print(f"    Successfully adjusted segment ID {segment_id_str} speed. Adjusted audio: {segment_adjusted_path}")
                adjusted_segment_files.append(segment_adjusted_path) # Store Path object directly
            
            except Exception as e_segment:
                print(f"    An error occurred during TTS or speed adjustment for segment ID {segment_id_str}: {e_segment}")
        
        # Merge all adjusted audio segments if any were successfully processed
        if adjusted_segment_files:
            print(f"  Merging {len(adjusted_segment_files)} adjusted audio segments...")
            _potential_final_dubbed_path = output_directory / f"{video_file.stem}_dubbed_voice_to_{safe_target_lang_code_for_filename}.wav"
            ffmpeg_concat_list_path = temp_segments_dir / "ffmpeg_concat_list.txt"
            
            list_file_created_successfully = False
            try:
                with open(ffmpeg_concat_list_path, "w", encoding='utf-8') as f_concat:
                    for audio_file_path in adjusted_segment_files: # Iterate over Path objects
                        # Write only the filename, as the list file is in the same directory as the audio files
                        f_concat.write(f"file '{audio_file_path.name}'\n")
                list_file_created_successfully = True
                print(f"    Successfully created ffmpeg concat list: {ffmpeg_concat_list_path}")
            except IOError as e_io:
                print(f"    Error creating ffmpeg concat list file ({ffmpeg_concat_list_path}): {e_io}")
            
            if list_file_created_successfully:
                merge_command = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0", 
                    "-i", str(ffmpeg_concat_list_path),
                    "-c", "copy", 
                    "-y", 
                    str(_potential_final_dubbed_path)
                ]
                try:
                    subprocess.run(merge_command, check=True, capture_output=True, text=True, encoding='utf-8')
                    print(f"  Successfully merged all adjusted audio segments into: {_potential_final_dubbed_path}")
                    final_dubbed_voice_track_path = _potential_final_dubbed_path # Assign on success
                    final_video_assembly_possible = True # Set flag on success
                    
                    # Optional: Clean up temporary segment files after successful merge
                    # print(f"  Cleaning up temporary audio segments directory: {temp_segments_dir}")
                    # shutil.rmtree(temp_segments_dir) # Uncomment to enable cleanup
                except subprocess.CalledProcessError as e_merge:
                    print(f"  Error during ffmpeg merging of audio segments: {e_merge}")
                    print(f"  FFmpeg stderr: {e_merge.stderr}")
                    # final_dubbed_voice_track_path remains None, final_video_assembly_possible remains False
                except FileNotFoundError:
                    print("  Error: ffmpeg command not found for merging audio segments.")
                    # final_dubbed_voice_track_path remains None, final_video_assembly_possible remains False
            else: # if list_file_created_successfully is False
                print(f"  Skipping ffmpeg merge because concat list file ({ffmpeg_concat_list_path}) could not be created.")
                # final_dubbed_voice_track_path remains None, final_video_assembly_possible remains False
        else: # if not adjusted_segment_files
            print("  No audio segments were successfully processed and speed-adjusted. Skipping final merge.")
            # final_dubbed_voice_track_path remains None, final_video_assembly_possible remains False
    elif not translated_segments_dict:
        print("Skipping TTS & Merging: No translated segments data available.")
        final_dubbed_voice_track_path = None
        final_video_assembly_possible = False
    elif not segments_for_translation:
         print("Skipping TTS & Merging: No original segments with timing information were prepared for translation.")
         final_dubbed_voice_track_path = None
         final_video_assembly_possible = False
    else: # Catch-all for other scenarios where previous steps didn't set these up
        final_dubbed_voice_track_path = None
        final_video_assembly_possible = False

    # Step 7: Final Video Assembly
    if final_video_assembly_possible and final_dubbed_voice_track_path and final_dubbed_voice_track_path.exists():
        print(f"\nStep 7: Assembling final video...")
        final_output_video_path = output_directory / f"{video_file.stem}_final_dubbed.mp4"

        ffmpeg_inputs = [
            "ffmpeg",
            "-i", str(video_file) # Input 0: Original video
        ]
        
        audio_input_count = 0
        audio_sources_for_filter = []

        # Input 1 (conditionally): Final dubbed voice track
        ffmpeg_inputs.extend(["-i", str(final_dubbed_voice_track_path)])
        dubbed_audio_input_index = 1 + audio_input_count # Starts at 1 (0 is video)
        audio_sources_for_filter.append(f"[{dubbed_audio_input_index}:a]")
        audio_input_count += 1

        # Input 2 (conditionally): Background audio
        background_audio_input_index = None
        if background_output_path.exists():
            ffmpeg_inputs.extend(["-i", str(background_output_path)])
            background_audio_input_index = 1 + audio_input_count
            audio_sources_for_filter.append(f"[{background_audio_input_index}:a]")
            audio_input_count += 1
            print(f"  Including background audio: {background_output_path}")
        else:
            print(f"  Background audio not found at {background_output_path}. Proceeding without it.")

        # Video mapping and codec
        maps_and_codecs = ["-map", "0:v", "-c:v", "copy"]

        # Audio mapping and codec
        if len(audio_sources_for_filter) > 1:
            # More than one audio source, so mix them
            filter_complex_str = f"{''.join(audio_sources_for_filter)}amix=inputs={len(audio_sources_for_filter)}:duration=longest[a_out]"
            maps_and_codecs.extend(["-filter_complex", filter_complex_str, "-map", "[a_out]"])
        elif len(audio_sources_for_filter) == 1:
            # Only one audio source (must be the dubbed voice in this logic branch)
            maps_and_codecs.extend(["-map", f"{dubbed_audio_input_index}:a"])
        else:
            # Should not happen if final_dubbed_voice_track_path exists, but as a fallback no new audio.
            print("  Warning: No audio sources identified for the final video. Video will be silent.")
            maps_and_codecs.extend(["-an"]) # No audio

        maps_and_codecs.extend(["-c:a", "aac", "-b:a", "192k"]) # Standard audio codec and bitrate
        maps_and_codecs.extend(["-y", str(final_output_video_path)]) # Overwrite and output path

        final_video_command = ffmpeg_inputs + maps_and_codecs

        print(f"  Running ffmpeg for final video assembly. Output: {final_output_video_path}")
        # print(f"    FFmpeg command: {' '.join(final_video_command)}") # For debugging
        try:
            process_result = subprocess.run(final_video_command, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"  Successfully assembled final dubbed video: {final_output_video_path}")
            if process_result.stdout:
                print(f"    FFmpeg stdout:\n{process_result.stdout}")
            if process_result.stderr:
                print(f"    FFmpeg stderr (may contain informational messages):\n{process_result.stderr}")
        except subprocess.CalledProcessError as e_video_assembly:
            print(f"  Error during final video assembly with ffmpeg: {e_video_assembly}")
            print(f"  FFmpeg command was: {' '.join(e_video_assembly.cmd)}")
            print(f"  FFmpeg stdout:\n{e_video_assembly.stdout}")
            print(f"  FFmpeg stderr:\n{e_video_assembly.stderr}")
        except FileNotFoundError:
            print("  Error: ffmpeg command not found for final video assembly.")
    elif not final_video_assembly_possible:
        print("\nStep 7: Final video assembly skipped because the dubbed voice track was not successfully created.")
    else: # Should not be reached if logic is correct, but for completeness
        print("\nStep 7: Final video assembly skipped due to missing dubbed voice track path.")

    print("\nAll processing steps complete.")

if __name__ == "__main__":
    main()