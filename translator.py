import os
from dotenv import load_dotenv # Import dotenv
from google import genai
from google.genai import types
import time # Added for retry delay

load_dotenv() # Load environment variables from .env file at the start

# Ensure the API key is set as an environment variable
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY" # User must set this externally

DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MAX_RETRIES = 2 # Maximum number of retries
RETRY_DELAY_SECONDS = 5 # Delay between retries

def translate_text_gemini(text_to_translate: str, 
                            target_language: str, 
                            source_language: str = None, 
                            model_name: str = DEFAULT_MODEL_NAME) -> str | None:
    """
    Translates the given text to the target language using the Gemini API.
    Loads GEMINI_API_KEY from .env file.
    Includes a retry mechanism for transient errors.

    Args:
        text_to_translate: The text to be translated.
        target_language: The language to translate the text into (e.g., "English", "Turkish").
        source_language: The source language of the text (e.g., "English", "Turkish"). 
                         If None, the model will attempt to auto-detect.
        model_name: The name of the Gemini model to use.

    Returns:
        The translated text as a string, or None if translation fails after retries.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it is set in your .env file or as an environment variable.")
        return None

    if source_language:
        prompt = f"Translate the following text accourding to context from {source_language} to {target_language}:\n\n{text_to_translate} and only output the translated text, no other text or comments. Output lenght should be similar to the original text. Gramatically correct."
    else:
        prompt = f"Translate the following text according to context to {target_language}:\n\n{text_to_translate} and only output the translated text, no other text or comments. Output lenght should be similar to the original text. Output should be gramatically correct."
    
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

    last_exception = None
    for attempt in range(MAX_RETRIES + 1): # MAX_RETRIES means we try initial + MAX_RETRIES more times
        try:
            client = genai.Client(api_key=api_key)
            full_translated_text = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                full_translated_text += chunk.text
            
            # Check if the result is non-empty and seems valid
            if full_translated_text and full_translated_text.strip():
                return full_translated_text.strip()
            else:
                # Treat empty or whitespace-only response as a failure for retry purposes
                raise ValueError("Received empty or whitespace-only translation from API.")

        except Exception as e:
            last_exception = e
            print(f"Error during Gemini translation (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Max retries reached. Translation failed. Last error: {last_exception}")
    
    # If loop finishes without returning, it means all retries failed
    # and last_exception will contain the exception from the final attempt.
    if last_exception: # Ensure we print it if it was set (i.e. at least one attempt failed)
        # This specific print might be redundant if the above print("Max retries reached...") already includes it.
        # However, if the loop completes for other reasons without success & last_exception is set, this could be useful.
        # For now, the primary usage is the modified print above.
        pass # last_exception is now used in the print statement above
    return None

if __name__ == '__main__':
    pass