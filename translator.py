import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import json

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 5

def translate_text_gemini(segments_to_translate: list[dict],
                            target_language: str,
                            source_language: str = None,
                            model_name: str = DEFAULT_MODEL_NAME) -> dict | None:

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it is set in your .env file or as an environment variable.")
        return None

    if not segments_to_translate:
        print("Translator Info: No segments provided for translation.")
        return {}

    input_data_for_gemini = {str(segment['id']): segment['text'] for segment in segments_to_translate}
    input_json_str = json.dumps(input_data_for_gemini, ensure_ascii=False, indent=2)

    source_lang_description = source_language if source_language else "the auto-detected source language"
    prompt = (
        f"You are an expert translation service. "
        f"I will provide you with a JSON object where keys are segment IDs and values are the text of those segments.\\n"
        f"Your task is to translate the text value for each segment ID from '{source_lang_description}' to '{target_language}'.\\n"
        f"You MUST return a single JSON object. This JSON object should have the exact same segment IDs as keys, and the corresponding translated text as values.\\n"
        f"For example, if the input is {{{{ \"0\": \"Hello world\", \"1\": \"How are you?\" }}}} and the target language is Turkish, "
        f"your output should be {{{{ \"0\": \"Merhaba dünya\", \"1\": \"Nasılsın?\" }}}}.\\n"
        f"Ensure the output is ONLY the JSON object and nothing else. Do not add any explanations or markdown formatting around the JSON.\\n\\n"
        f"Input JSON:\\n"
        f"{input_json_str}"
    )

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
        response_mime_type="application/json",
    )

    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=api_key)
            
            response_chunks = []
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                response_chunks.append(chunk)

            full_response_text = "".join(chunk.text for chunk in response_chunks if hasattr(chunk, 'text') and chunk.text)

            if full_response_text and full_response_text.strip():
                translated_data = json.loads(full_response_text)
                return translated_data
            else:
                print(f"Gemini Info: Received empty or whitespace-only response text when expecting JSON (Attempt {attempt + 1}).")

        except json.JSONDecodeError as json_e:
            last_exception = json_e
            print(f"Error decoding JSON response from Gemini (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {json_e}")
            print(f"Gemini Raw Response Text: '{full_response_text}'")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Max retries reached for JSON decoding. Translation failed. Last error: {last_exception}")
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