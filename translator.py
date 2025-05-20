import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 5

def translate_text_gemini(text_to_translate: str,
                            target_language: str,
                            source_language: str = None,
                            model_name: str = DEFAULT_MODEL_NAME) -> str | None:

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it is set in your .env file or as an environment variable.")
        return None

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

    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=api_key)
            full_translated_text = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                full_translated_text += chunk.text

            if full_translated_text and full_translated_text.strip():
                return full_translated_text.strip()
            else:
                raise ValueError("Received empty or whitespace-only translation from API.")

        except Exception as e:
            last_exception = e
            print(f"Error during Gemini translation (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Max retries reached. Translation failed. Last error: {last_exception}")

    if last_exception:
        pass
    return None

if __name__ == '__main__':
    pass