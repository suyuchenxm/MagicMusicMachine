import base64
import json
import os

import anthropic
import gradio as gr

# Remember to put your API Key here
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# image1_url = "https://i.abcnewsfe.com/a/7d849ccc-e0fe-4416-959d-85889e338add/dune-1-ht-bb-231212_1702405287482_hpMain_16x9.jpeg"
image1_media_type = "image/jpeg"
# image1_data = base64.b64encode(httpx.get(image1_url).content).decode("utf-8")
#

SYSTEM_PROMPT = """You are an expert llm prompt engineer, you understand the structure of llms and facebook musicgen text to audio model. You will be provided with an image, and require to output a prompt for the musicgen model to capture the essense of the image. Try to do it step by step, evaluate and analyze the image thoroughly. After that, develop a prompt that contains music genera, style, instrument, and all the other details needed. This prompt will be provided to musicgen model to generate a 15s audio clip.

Here are some descriptions from musicgen model:
The model was trained with descriptions from a stock music catalog, descriptions that will work best should include some level of detail on the instruments present, along with some intended use case (e.g. adding “perfect for a commercial” can somehow help).

Try to make the prompt simple and concise with only 1-2 sentences

Make sure the ouput is in JSON fomat, with two items `description` and `prompt`"""

SYSTEM_PROMPT_AUDIO = """You are an expert llm prompt engineer, you understand the structure of llms and facebook musicgen text to audio model. You will be provided with an image, and require to output a prompt for the musicgen model to capture the essense of the image. Try to do it step by step, evaluate and analyze the image thoroughly. After that, develop a prompt that contains the detail of what background sounds this image should have. This prompt will be provided to audiogen model to generate a 15s audio clip.
Try to make the prompt simple and concise with only 1-2 sentences

Make sure the ouput is in JSON fomat, with two items `description` and `prompt`
"""

PROMPT_IMPROVEMENT_GENERATE_PROMPT = """
You are an export llm prompt enginner, you will be helping the user to improve their prompts. here are some examples of good prompts
- "90s rock song with electric guitar and heavy drums"
- "An 80s driving pop song with heavy drums and synth pads in the background"
- "An energetic hip-hop music piece, with synth sounds and strong bass. There is a rhythmic hi-hat patten in the drums."
- "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle."
- "Classic reggae track with an electronic guitar solo"

You will be provided with a prompt and you need to improve it. Make sure the prompt is simple and concise with only 1-2 sentences. The output should be in JSON format, with one item `prompt`
"""


def improve_prompt(prompt):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system=PROMPT_IMPROVEMENT_GENERATE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    message_object = json.loads(message.content[0].text)
    prompt = message_object["prompt"]
    return message_object, prompt


def generate_caption(image_file, model_file, progress=gr.Progress()):
    if model_file == "facebook/audiogen-medium":
        system_prompt = SYSTEM_PROMPT_AUDIO
    else:
        system_prompt = SYSTEM_PROMPT
    with open(image_file, "rb") as f:
        image_encoded = base64.b64encode(f.read()).decode("utf-8")
    progress(0, desc="Starting image captioning...")
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image1_media_type,
                            "data": image_encoded,
                        },
                    },
                    {"type": "text", "text": "develop the prompt based on this image"},
                ],
            }
        ],
    )
    progress(100, desc="image captioning...Done!")
    # Parse the content string into a Python object
    message_object = json.loads(message.content[0].text)
    # Access the description and prompt from the message object
    description = message_object["description"]
    prompt = message_object["prompt"]
    print(description)
    print(prompt)
    return message_object, description, prompt
