import whisper

def transcribe_audio_whisper(audio_path_str: str, model_name: str = "base", device: str = "cuda", language: str = None):
    try:
        model = whisper.load_model(model_name, device=device)
        
        transcribe_options = {"fp16": False}
        if language:
            transcribe_options["language"] = language
            print(f"Whisper Info: Transcribing with specified language: {language}")
        else:
            print("Whisper Info: Transcribing with language auto-detection.")

        result = model.transcribe(audio_path_str, **transcribe_options)
        return result
    except Exception as e:
        print(f"Error during Whisper transcription (model: {model_name}, device: {device}, language: {language}): {e}")
        return None