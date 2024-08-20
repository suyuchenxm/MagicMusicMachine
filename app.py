import os
import gradio as gr
from audiocraft.models import MAGNeT, MusicGen, AudioGen

# from gradio_components.image import generate_caption, improve_prompt
from gradio_components.prediction import predict, transcribe

import re
import argparse
from gradio_components.model_cards import TEXT_TO_MIDI_MODELS, TEXT_TO_SOUND_MODELS, TEXT_TO_MUSIC_MODELS, MODEL_CARDS, MELODY_CONDITIONED_MODELS

def load_model(version='facebook/musicgen-large', test=False):
    global MODEL
    if test:
        return MODEL
    if MODEL is None or MODEL.name != version:
        del MODEL
        MODEL = None  # in case loading would crash
    
    print("Loading model", version)
    if re.match(r"magnet", version):
        MODEL = MAGNeT.get_pretrained(version)
    elif re.match(r"musicgen", version):
        MODEL = MusicGen.get_pretrained(version)
    elif re.match(r"musiclang", version):
        # TODO: Implement MusicLang
        pass
    elif re.match(r"audiogen", version):
        MODEL = AudioGen.get_pretrained('facebook/audiogen-medium')
    else:
        raise ValueError("Invalid model version")
    
    return MODEL
    

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


def toggle_melody_condition(melody_condition):
    if melody_condition:
        return gr.Audio(
            sources=["microphone", "upload"],
            label="Record or upload your audio",
            show_label=True,
            visible=True,
        )
    else:
        return gr.Audio(
            sources=["microphone", "upload"],
            label="Record or upload your audio",
            show_label=True,
            visible=False,
        )


def toggle_custom_prompt(customize, difficulty, style):
    if customize:
        return gr.Textbox(label="Type your prompt", interactive=True, visible=True)
    else:
        prompt = generate_prompt(difficulty, style)
        return gr.Textbox(
            label="Generated Prompt", value=prompt, interactive=False, visible=True
        )


def show_caption(show_caption_condition, description, prompt):
    if show_caption_condition:
        return (
            gr.Textbox(
                label="Image Caption",
                value=description,
                interactive=False,
                show_label=True,
                visible=True,
            ),
            gr.Textbox(
                label="Generated Prompt",
                value=prompt,
                interactive=True,
                show_label=True,
                visible=True,
            ),
            gr.Button("Generate Music", interactive=True, visible=True),
        )
    else:
        return (
            gr.Textbox(
                label="Image Caption",
                value=description,
                interactive=False,
                show_label=True,
                visible=False,
            ),
            gr.Textbox(
                label="Generated Prompt",
                value=prompt,
                interactive=True,
                show_label=True,
                visible=False,
            ),
            gr.Button(label="Generate Music", interactive=True, visible=True),
        )


def optimize_fn(prompt):
    message_object, prompt = improve_prompt(prompt)
    return prompt


def display_prompt(prompt):
    return gr.Textbox(
        label="Generated Prompt", value=prompt, interactive=False, visible=True
    )


def post_submit(show_caption, model_path, image_input):
    _, description, prompt = generate_caption(image_input, model_path)
    return (
        gr.Textbox(
            label="Image Caption",
            value=description,
            interactive=False,
            show_label=True,
            visible=show_caption,
        ),
        gr.Textbox(
            label="Generated Prompt",
            value=prompt,
            interactive=True,
            show_label=True,
            visible=show_caption,
        ),
        gr.Button("Generate Music", interactive=True, visible=True),
    )

MODEL = None

def UI(share=False):
    with gr.Blocks() as demo:
        with gr.Tab("Generate Music by text"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_path = gr.Dropdown(
                            choices=TEXT_TO_MUSIC_MODELS,
                            label="Select the model",
                            value="facebook/musicgen-melody-large",
                        )
                        MODEL = load_model(model_path)

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

                    config_output_textbox = gr.Textbox(label="Model Configs", render=False)
                    @gr.render(inputs=model_path, triggers=[model_path.change])
                    def show_config_options(model_path):
                        print(model_path)
                        
                        with gr.Accordion("Model Generation Configs"):
                            if "magnet" in model_path:
                                with gr.Row():
                                    duration = gr.Slider(
                                        minimum=10,
                                        maximum=60,
                                        value=10,
                                        label="Duration",
                                        interactive=True,
                                    )
                                    top_k = gr.Number(label="Top-k", value=0, interactive=True)
                                    top_p = gr.Number(label="Top-p", value=0.95, interactive=True)
                                    temperature = gr.Number(
                                        label="Temperature", value=1.0, interactive=True
                                    )
                                    span_arrangement = gr.Radio(["nonoverlap", "stride1"], value='nonoverlap', label="span arrangment", info=" Use either non-overlapping spans ('nonoverlap') or overlapping spans ('stride1') ")
                                @gr.on(inputs=[duration, top_k, top_p, temperature, span_arrangement], outputs=config_output_textbox)
                                def return_model_configs(duration, top_k, top_p, temperature, span_arrangement):
                                    return {"duration": duration, "top_k": top_k, "top_p": top_p, "temperature": temperature, "span_arrangement": span_arrangement}
                            
                            else:
                                with gr.Row():
                                    duration = gr.Slider(
                                        minimum=10,
                                        maximum=60,
                                        value=10,
                                        label="Duration",
                                        interactive=True,
                                    )
                                    use_sampling = gr.Checkbox(label="Use Sampling", interactive=True, value=True)
                                    topk = gr.Number(label="Top-k", value=0, interactive=True)
                                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                                    temperature = gr.Number(
                                        label="Temperature", value=1.0, interactive=True
                                    )
                                @gr.on(inputs=[duration, use_sampling, topk, topp, temperature], outputs=config_output_textbox)
                                def return_model_configs(duration, use_sampling, topk, topp, temperature):
                                    return {"duration": duration, "use_sampling": use_sampling, "topk": topk, "topp": topp, "temperature": temperature}

                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Generate Music")
                        output_audio = gr.Audio(
                            "listen to the generated music", type="filepath", visible=False
                        )
                        output_waveform = gr.Video(label="Generated Music")
                        output_audio = gr.Audio(label="Generated Music (wav)", type='filepath')
                        gr.on(
                            triggers=[submit.click],
                            inputs=[model_path, config_output_textbox, text_prompt, text_prompt, num_outputs], 
                            outputs=[output_audio, output_waveform],
                            fn=predict
                            )
                        
                    with gr.Row():
                        transcribe_button = gr.Button("Transcribe")
                        d = gr.DownloadButton("Download the file", visible=False)
                        transcribe_button.click(
                            transcribe, inputs=[output_audio], outputs=d
                        )

                # with gr.Column():
                #     with gr.Row():
                #         melody = gr.Audio(
                #             sources=["microphone", "upload"],
                #             label="Record or upload your audio",
                #             # interactive=True,
                #             show_label=True,
                #         )
                #     with gr.Row():
                #         submit = gr.Button("Generate Music")
                #         output_audio = gr.Audio(
                #             "listen to the generated music", type="filepath"
                #         )
                #     with gr.Row():
                #         transcribe_button = gr.Button("Transcribe")
                #         d = gr.DownloadButton("Download the file", visible=False)
                #         transcribe_button.click(
                #             transcribe, inputs=[output_audio], outputs=d
                #         )

            
        # with gr.Tab("Generate Music by melody"):
            # submit.click(
            #     fn=predict,
            #     inputs=[
            #         model_path,
            #         prompt,
            #         melody,
            #         duration,
            #         topk,
            #         topp,
            #         temperature,
            #         sample_rate,
            #     ],
            #     outputs=output_audio,
            # )
            # gr.Examples(
            #     examples=[
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/twinkle_twinkle_little_stars_mozart_20sec"
            #                 ".mp3",
            #             ),
            #             "Easy",
            #             32000,
            #             20,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/golden_hour_20sec.mp3",
            #             ),
            #             "Easy",
            #             32000,
            #             20,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/turkish_march_mozart_20sec.mp3",
            #             ),
            #             "Easy",
            #             32000,
            #             20,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/golden_hour_20sec.mp3",
            #             ),
            #             "Hard",
            #             32000,
            #             20,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/golden_hour_20sec.mp3",
            #             ),
            #             "Hard",
            #             32000,
            #             40,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/golden_hour_20sec.mp3",
            #             ),
            #             "Hard",
            #             16000,
            #             20,
            #         ],
            #         [
            #             os.path.join(
            #                 os.path.dirname(__file__),
            #                 "./data/audio/old_town_road20sec.mp3",
            #                 ),
            #             "Hard",
            #             32000,
            #             40,
            #             ],
            #     ],
            #     inputs=[melody, difficulty, sample_rate, duration],
            #     label="Audio Examples",
            #     outputs=[output_audio],
                # cache_examples=True,
            # )
        # with gr.Tab("Generate Music by image"):
        #     with gr.Row():
        #         with gr.Column():
        #             image_input = gr.Image("Upload an image", type="filepath")
        #             melody_condition = gr.Checkbox(
        #                 label="Generate music by melody", interactive=True, value=False
        #             )
        #             melody = gr.Audio(
        #                 sources=["microphone", "upload"],
        #                 label="Record or upload your audio",
        #                 show_label=True,
        #                 visible=False,
        #             )
        #             melody_condition.change(
        #                 fn=toggle_melody_condition,
        #                 inputs=[melody_condition],
        #                 outputs=melody,
        #             )
        #             description = gr.Textbox(
        #                 label="Image Captioning",
        #                 show_label=True,
        #                 interactive=False,
        #                 visible=False,
        #             )
        #             prompt = gr.Textbox(
        #                 label="Generated Prompt",
        #                 show_label=True,
        #                 interactive=True,
        #                 visible=False,
        #             )
        #             show_prompt = gr.Checkbox(label="Show the prompt", interactive=True)
        #             submit = gr.Button("submit", interactive=True, visible=True)
        #             generate = gr.Button(
        #                 "Generate Music", interactive=True, visible=False
        #             )

        #         with gr.Column():
        #             with gr.Row():
        #                 model_path = gr.Dropdown(
        #                     choices=TEXT_TO_MUSIC_MODELS + TEXT_TO_SOUND_MODELS + MELODY_CONDITIONED_MODELS,
        #                     label="Select the model",
        #                     value="facebook/musicgen-large",
        #                 )
        #             with gr.Row():
        #                 duration = gr.Slider(
        #                     minimum=10,
        #                     maximum=60,
        #                     value=10,
        #                     label="Duration",
        #                     interactive=True,
        #                 )
        #             topk = gr.Number(label="Top-k", value=250, interactive=True)
        #             topp = gr.Number(label="Top-p", value=0, interactive=True)
        #             temperature = gr.Number(
        #                 label="Temperature", value=1.0, interactive=True
        #             )
        #             sample_rate = gr.Number(
        #                 label="output music sample rate", value=32000, interactive=True
        #             )
        #         with gr.Column():
        #             output_audio = gr.Audio(
        #                 "listen to the generated music",
        #                 type="filepath",
        #                 show_label=True,
        #             )
        #             transcribe_button = gr.Button("Transcribe")
        #             d = gr.DownloadButton("Download the file", visible=False)
        #     submit.click(
        #         fn=post_submit,
        #         inputs=[show_prompt, model_path, image_input],
        #         outputs=[description, prompt, generate],
        #     )
        #     show_prompt.change(
        #         fn=show_caption,
        #         inputs=[show_prompt, description, prompt],
        #         outputs=[description, prompt, generate],
        #     )
        #     transcribe_button.click(transcribe, inputs=[output_audio], outputs=d)
        #     generate.click(
        #         fn=predict,
        #         inputs=[
        #             model_path,
        #             prompt,
        #             melody,
        #             duration,
        #             topk,
        #             topp,
        #             temperature,
        #             sample_rate,
        #         ],
        #         outputs=output_audio,
        #     )

        #     gr.Examples(
        #         examples=[
        #             [
        #                 os.path.join(
        #                     os.path.dirname(__file__),
        #                     "./data/image/kids_drawing.jpeg",
        #                 ),
        #                 False,
        #                 None,
        #                 "facebook/musicgen-large",
        #             ],
        #             [
        #                 os.path.join(
        #                     os.path.dirname(__file__),
        #                     "./data/image/cat.jpeg",
        #                 ),
        #                 False,
        #                 None,
        #                 "facebook/musicgen-large",
        #             ],
        #             [
        #                 os.path.join(
        #                     os.path.dirname(__file__),
        #                     "./data/image/cat.jpeg",
        #                 ),
        #                 True,
        #                 "./data/audio/the_nutcracker_dance_of_the_reed_flutes.mp3",
        #                 "facebook/musicgen-melody-large",
        #             ],
        #             [
        #                 os.path.join(
        #                     os.path.dirname(__file__),
        #                     "./data/image/beach.jpeg",
        #                 ),
        #                 False,
        #                 None,
        #                 "facebook/audiogen-medium",
        #             ],
        #         ],
        #         inputs=[image_input, melody_condition, melody, model_path],
        #         label="Audio Examples",
        #         outputs=[output_audio],
        #         # cache_examples=True,
        #     )

    demo.queue().launch(share=share)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help='Enable sharing.')
    args = parser.parse_args()

    UI(share=args.share)
