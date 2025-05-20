import os
from dotenv import load_dotenv # Import dotenv
from google import genai
from google.genai import types

load_dotenv() # Load environment variables from .env file at the start

# Ensure the API key is set as an environment variable
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY" # User must set this externally

DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"

def translate_text_gemini(text_to_translate: str, 
                            target_language: str, 
                            source_language: str = None, 
                            model_name: str = DEFAULT_MODEL_NAME) -> str | None:
    """
    Translates the given text to the target language using the Gemini API.
    Loads GEMINI_API_KEY from .env file.

    Args:
        text_to_translate: The text to be translated.
        target_language: The language to translate the text into (e.g., "English", "Turkish").
        source_language: The source language of the text (e.g., "English", "Turkish"). 
                         If None, the model will attempt to auto-detect.
        model_name: The name of the Gemini model to use.

    Returns:
        The translated text as a string, or None if translation fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it is set in your .env file or as an environment variable.")
        return None

    try:
        client = genai.Client(api_key=api_key)

        if source_language:
            prompt = f"Translate the following text accourding to context from {source_language} to {target_language}:\n\n{text_to_translate} and only output the translated text, no other text or comments. Output lenght should be similar to the original text."
        else:
            prompt = f"Translate the following text according to context to {target_language}:\n\n{text_to_translate} and only output the translated text, no other text or comments. Output lenght should be similar to the original text."
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0.5,
            response_mime_type="text/plain",
        )

        full_translated_text = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            full_translated_text += chunk.text
        
        return full_translated_text.strip()

    except Exception as e:
        print(f"Error during Gemini translation: {e}")
        # More specific error handling can be added based on google.generativeai.client. generación_error types
        # For example, if e.response.prompt_feedback.block_reason:
        # print(f"Translation blocked. Reason: {e.response.prompt_feedback.block_reason}")
        return None

if __name__ == '__main__':
    # This is for testing the translator.py script directly.
    # Make sure GEMINI_API_KEY is set in your environment.
    print("Testing Gemini Translator (ensure .env file with GEMINI_API_KEY is present)...")
    test_text_tr = "Merhaba dünya, bu bir test çevirisidir."
    test_text_en = "Hello world, this is a test translation."
    target_lang_en = "English"
    target_lang_tr = "Turkish"

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not loaded (checked after load_dotenv). Ensure .env file is correct. Skipping direct test.")
    else:
        print(f"\nOriginal (TR): {test_text_tr}")
        translated_to_en = translate_text_gemini(test_text_tr, target_lang_en, source_language="Turkish")
        if translated_to_en:
            print(f"Translated to {target_lang_en}: {translated_to_en}")
        else:
            print(f"Translation to {target_lang_en} failed.")

        print(f"\nOriginal (EN): {test_text_en}")
        translated_to_tr = translate_text_gemini(test_text_en, target_lang_tr, source_language="English")
        if translated_to_tr:
            print(f"Translated to {target_lang_tr}: {translated_to_tr}")
        else:
            print(f"Translation to {target_lang_tr} failed.") 