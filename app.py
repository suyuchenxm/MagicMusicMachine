import os
import gradio as gr
from audiocraft.models import MAGNeT, MusicGen, AudioGen

# from gradio_components.image import generate_caption, improve_prompt
from gradio_components.image import generate_caption_gpt4
from gradio_components.prediction import predict, transcribe

import re
import argparse
from gradio_components.model_cards import TEXT_TO_MIDI_MODELS, TEXT_TO_SOUND_MODELS, MELODY_CONTINUATION_MODELS, TEXT_TO_MUSIC_MODELS, MODEL_CARDS, MELODY_CONDITIONED_MODELS
import ast
import json


    

theme = gr.themes.Glass(
    primary_hue="fuchsia",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Source Sans Pro"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
).set(
    body_background_fill_dark="*background_fill_primary",
    embed_radius="*table_radius",
    background_fill_primary="*neutral_50",
    background_fill_primary_dark="*neutral_950",
    background_fill_secondary_dark="*neutral_900",
    border_color_accent="*neutral_600",
    border_color_accent_subdued="*color_accent",
    border_color_primary_dark="*neutral_700",
    block_background_fill="*background_fill_primary",
    block_background_fill_dark="*neutral_800",
    block_border_width="1px",
    block_label_background_fill="*background_fill_primary",
    block_label_background_fill_dark="*background_fill_secondary",
    block_label_text_color="*neutral_500",
    block_label_text_size="*text_sm",
    block_label_text_weight="400",
    block_shadow="none",
    block_shadow_dark="none",
    block_title_text_color="*neutral_500",
    block_title_text_weight="400",
    panel_border_width="0",
    panel_border_width_dark="0",
    checkbox_background_color_dark="*neutral_800",
    checkbox_border_width="*input_border_width",
    checkbox_label_border_width="*input_border_width",
    input_background_fill="*neutral_100",
    input_background_fill_dark="*neutral_700",
    input_border_color_focus_dark="*neutral_700",
    input_border_width="0px",
    input_border_width_dark="0px",
    slider_color="#2563eb",
    slider_color_dark="#2563eb",
    table_even_background_fill_dark="*neutral_950",
    table_odd_background_fill_dark="*neutral_900",
    button_border_width="*input_border_width",
    button_shadow_active="none",
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_dark="*primary_700",
    button_primary_background_fill_hover="*button_primary_background_fill",
    button_primary_background_fill_hover_dark="*button_primary_background_fill",
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_dark="*neutral_600",
    button_secondary_background_fill_hover="*button_secondary_background_fill",
    button_secondary_background_fill_hover_dark="*button_secondary_background_fill",
    button_cancel_background_fill="*button_secondary_background_fill",
    button_cancel_background_fill_dark="*button_secondary_background_fill",
    button_cancel_background_fill_hover="*button_cancel_background_fill",
    button_cancel_background_fill_hover_dark="*button_cancel_background_fill",
)



def generate_prompt(prompt, style):
    prompt = ','.join([prompt]+style)
    return prompt


def UI(share=False):
    with gr.Blocks() as demo:
        with gr.Tab("Generate Music by text"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_path = gr.Dropdown(
                            choices=TEXT_TO_MUSIC_MODELS,
                            label="Select the model",
                            value="facebook/musicgen-large",
                        )

                    with gr.Row():
                        text_prompt = gr.Textbox(
                            label="Let's make a song about ...",
                            value="First day learning music generation in Standford university",
                            interactive=True,
                            visible=True,
                        )
                        num_outputs = gr.Number(
                            label="Number of outputs",
                            value=1,
                            minimum=1,
                            maximum=10,
                            interactive=True,
                        )
                   
                    with gr.Row():
                        style = gr.CheckboxGroup(
                            ["Jazz", "Classical Music", "Hip Hop", "Ragga Jungle", "Dark Jazz", "Soul", "Blues", "80s Rock N Roll"],
                            value=None,
                            label="music genre",
                            interactive=True,
                        )
                        @gr.on(inputs=[style], outputs=text_prompt)
                        def update_prompt(style):
                            return generate_prompt(text_prompt.value, style)

                    config_output_textbox = gr.Textbox(label="Model Configs", visible=False)
                    
                    @gr.render(inputs=model_path)
                    def show_config_options(model_path):
                        print(model_path)
                        
                        with gr.Accordion("Model Generation Configs"):
                            if "magnet" in model_path:
                                with gr.Row():
                                    top_k = gr.Number(label="Top-k", value=300, interactive=True)
                                    top_p = gr.Number(label="Top-p", value=0, interactive=True)
                                    temperature = gr.Number(
                                        label="Temperature", value=1.0, interactive=True
                                    )
                                    span_arrangement = gr.Radio(["nonoverlap", "stride1"], value='nonoverlap', label="span arrangment", info=" Use either non-overlapping spans ('nonoverlap') or overlapping spans ('stride1') ")
                                @gr.on(inputs=[top_k, top_p, temperature, span_arrangement], outputs=config_output_textbox)
                                def return_model_configs(top_k, top_p, temperature, span_arrangement):
                                    return {"top_k": top_k, "top_p": top_p, "temperature": temperature, "span_arrangement": span_arrangement}
                            else:
                                with gr.Row():
                                    duration = gr.Slider(
                                        minimum=10,
                                        maximum=30,
                                        value=30,
                                        label="Duration",
                                        interactive=True,
                                    )
                                    use_sampling = gr.Checkbox(label="Use Sampling", interactive=True, value=True)
                                    top_k = gr.Number(label="Top-k", value=300, interactive=True)
                                    top_p = gr.Number(label="Top-p", value=0, interactive=True)
                                    temperature = gr.Number(
                                        label="Temperature", value=1.0, interactive=True
                                    )
                                @gr.on(inputs=[duration, use_sampling, top_k, top_p, temperature], outputs=config_output_textbox)
                                def return_model_configs(duration, use_sampling, top_k, top_p, temperature):
                                    return {"duration": duration, "use_sampling": use_sampling, "top_k": top_k, "top_p": top_p, "temperature": temperature}

                with gr.Column():
                    with gr.Row():
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                        interactive=True, elem_id="melody-input", visible=False)
                        submit = gr.Button("Generate Music")
                    result_text = gr.Textbox(label="Generated Music (text)", type="text", interactive=False)
                    print(result_text)
                    output_audios = []
                    @gr.render(inputs=result_text)
                    def show_output_audio(tmp_paths):
                        if tmp_paths:
                            tmp_paths = ast.literal_eval(tmp_paths)
                            print(tmp_paths)
                            for i in range(len(tmp_paths)):
                                tmp_path = tmp_paths[i]
                                _audio = gr.Audio(value=tmp_path , label=f"Generated Music {i}", type='filepath', interactive=False, visible=True)
                                output_audios.append(_audio)
                
                    submit.click(
                        fn=predict,
                        inputs=[model_path, config_output_textbox, text_prompt, melody, num_outputs], 
                        outputs=result_text,
                        queue=True
                        )
                
            
        with gr.Tab("Generate Music by melody"):
            with gr.Column():
                with gr.Row():
                    radio_melody_condition = gr.Radio(["Muisc Continuation", "Music Conditioning"], value=None, label="Select the condition")
                    model_path2 = gr.Dropdown(label="model")
                    @gr.on(inputs=radio_melody_condition, outputs=model_path2)
                    def model_selection(radio_melody_condition):
                        if radio_melody_condition == "Muisc Continuation":
                            model_path2 = gr.Dropdown(
                                choices=MELODY_CONTINUATION_MODELS,
                                label="Select the model",
                                value="facebook/musicgen-large",
                                interactive=True,
                                visible=True
                            )
                        elif radio_melody_condition == "Music Conditioning":
                            model_path2 = gr.Dropdown(
                                choices=MELODY_CONDITIONED_MODELS,
                                label="Select the model",
                                value="facebook/musicgen-melody-large",
                                interactive=True,
                                visible=True
                            )
                        else:
                            model_path2 = gr.Dropdown(
                                choices=TEXT_TO_SOUND_MODELS,
                                label="Select the model",
                                value="facebook/musicgen-large",
                                interactive=True,
                                visible=False
                            )
                        return model_path2
                    upload_melody = gr.Audio(sources=["upload", "microphone"], type="filepath", label="File")
                    prompt_text2 = gr.Textbox(
                        label="Let's make a song about ...",
                        value=None,
                        interactive=True,
                        visible=True,
                    )
                with gr.Row():
                    config_output_textbox2 = gr.Textbox(
                        label="Model Configs", 
                        visible=True)
                    with gr.Row():
                        duration2 = gr.Number(10, label="Duration", interactive=True)
                        num_outputs2 = gr.Number(1, label="Number of outputs", interactive=True)

                    @gr.on(inputs=[duration2], outputs=config_output_textbox2)
                    def return_model_configs2(duration):
                        return {"duration": duration, "use_sampling": True, "top_k": 300, "top_p": 0, "temperature": 1}
                    submit2 = gr.Button("Generate Music")
                    result_text2 = gr.Textbox(label="Generated Music (melody)", type="text", interactive=False, visible=True)
                    submit2.click(
                        fn=predict,
                        inputs=[model_path2, config_output_textbox2, prompt_text2, upload_melody, num_outputs2],
                        outputs=result_text2,
                        queue=True
                    )

                    @gr.render(inputs=result_text2)
                    def show_output_audio(tmp_paths):
                        if tmp_paths:
                            tmp_paths = ast.literal_eval(tmp_paths)
                            print(tmp_paths)
                            for i in range(len(tmp_paths)):
                                tmp_path = tmp_paths[i]
                                _audio = gr.Audio(value=tmp_path , label=f"Generated Music {i}", type='filepath', interactive=False)
                                output_audios.append(_audio)
            gr.Examples(
                examples = [
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/audio/Suri's Improv.mp3"
                        ),
                        30, 
                        "facebook/musicgen-large",
                        "Muisc Continuation",
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/audio/lie_no_tomorrow_20sec.wav"
                        ),
                        40, 
                        "facebook/musicgen-melody-large",
                        "Music Conditioning",
                    ]
                ],
                inputs=[upload_melody, duration2, model_path2, radio_melody_condition],
            )

        with gr.Tab("Generate Music by image"):
            with gr.Column():
                with gr.Row():
                    image_input = gr.Image("Upload an image", type="filepath")
                    with gr.Accordion("Image Captioning", open=False):
                        image_description = gr.Textbox(label='image description', visible=True, interactive=False)
                        image_caption = gr.Textbox(label='generated text prompt', visible=True, interactive=True)
                    @gr.on(inputs=image_input, outputs=[image_description, image_caption])
                    def generate_image_text_prompt(image_input):
                        if image_input:
                            image_description, image_caption = generate_caption_gpt4(image_input, model_path)
                            # meesage_object, description, prompt = generate_caption_claude3(image_input, model_path)
                            return image_description, image_caption
                        return "", ""
                with gr.Row():
                    melody3 = gr.Audio(sources=["upload", "microphone"], type="filepath", label="File", visible=True)
            with gr.Column():
                model_path3 = gr.Dropdown(
                    choices=TEXT_TO_SOUND_MODELS + TEXT_TO_MUSIC_MODELS + MELODY_CONDITIONED_MODELS,
                    label="Select the model",
                    value="facebook/musicgen-large",
                )
                duration3 = gr.Number(30, visible=False, label="Duration")
                submit3 = gr.Button("Generate Music")
                result_text3 = gr.Textbox(label="Generated Music (image)", type="text", interactive=False, visible=True)
                def predict_image_music(model_path3, image_caption, duration3, melody3):
                    model_configs = {"duration": duration3, "use_sampling": True, "top_k": 250, "top_p": 0, "temperature": 1}
                    return predict(
                        model_version = model_path3, 
                        generation_configs = model_configs, 
                        prompt_text = image_caption, 
                        prompt_wav = melody3
                        )

                submit3.click(
                    fn=predict_image_music,
                    inputs=[model_path3, image_caption, duration3, melody3],
                    outputs=result_text3,
                    queue=True
                )

                @gr.render(inputs=result_text3)
                def show_output_audio(tmp_paths):
                    if tmp_paths:
                        tmp_paths = ast.literal_eval(tmp_paths)
                        print(tmp_paths)
                        for i in range(len(tmp_paths)):
                            tmp_path = tmp_paths[i]
                            _audio = gr.Audio(value=tmp_path , label=f"Generated Music {i}", type='filepath', interactive=False)
                            output_audios.append(_audio)

                @gr.render(inputs=result_text3)
                def show_transcribt_audio(tmp_paths):
                    transcribe(tmp_paths)
            gr.Examples(
                examples = [
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/image/beach.jpeg"
                        ),
                        "facebook/musicgen-large",
                        30,
                        None,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/image/beach.jpeg"
                        ),
                        "facebook/audiogen-medium",
                        15,
                        None,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/image/beach.jpeg"
                        ),
                        "facebook/musicgen-melody-large",
                        30,
                        os.path.join(
                            os.path.dirname(__file__), "./data/audio/Suri's Improv.mp3"
                        ),
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__), "./data/image/cat.jpeg"
                        ),
                        "facebook/musicgen-large",
                        30,
                        None,
                    ],
                ],
                inputs=[image_input, model_path3, duration3, melody3],
            )

    demo.queue().launch(share=share)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help='Enable sharing.')
    args = parser.parse_args()

    UI(share=args.share)
