import whisper

def transcribe_audio_whisper(audio_path_str: str, model_name: str = "base", device: str = "cuda", language: str = None):
    """
    Transcribes the given audio file using OpenAI Whisper.

    Args:
        audio_path_str: Path to the audio file (e.g., vocals.wav).
        model_name: Name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
        device: The device to use for computation ("cuda" or "cpu").
        language: The language of the audio. If None, Whisper will attempt to auto-detect.

    Returns:
        A dictionary containing the transcription results (e.g., text, segments, language).
        Returns None if transcription fails.
    """
    try:
        # Let whisper handle device selection if torch.cuda.is_available() and device is "cuda"
        # Whisper might internally fallback to CPU if CUDA is requested but not truly usable for it.
        model = whisper.load_model(model_name, device=device)
        
        # Determine fp16 based on device. Typically, fp16 is beneficial on CUDA but not CPU.
        # Whisper itself might also manage this, but explicit control can be useful.
        use_fp16 = (device == "cuda") # Default to True for CUDA, False for CPU
        # However, we previously set fp16=False for broader compatibility. Let's stick to that for now
        # unless specific performance tuning is required.
        # To make it strictly conditional: result = model.transcribe(audio_path_str, fp16=use_fp16)
        
        # Transcribe, passing the language parameter if provided.
        # If language is None, Whisper will perform auto-detection.
        transcribe_options = {"fp16": False} # Keeping fp16=False for general compatibility
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