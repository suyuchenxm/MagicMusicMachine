import os

import gradio as gr

from gradio_components.image import generate_caption, improve_prompt
from gradio_components.prediction import predict, transcribe

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


_AUDIOCRAFT_MODELS = [
    "facebook/musicgen-melody",
    "facebook/musicgen-medium",
    "facebook/musicgen-small",
    "facebook/musicgen-large",
    "facebook/musicgen-melody-large",
    "facebook/audiogen-medium",
]


def generate_prompt(difficulty, style):
    _DIFFICULTY_MAPPIN = {
        "Easy": "beginner player",
        "Medium": "player who has 2-3 years experience",
        "Hard": "player who has more than 4 years experiences",
    }
    prompt = "piano only music for a {} to practice with the touch of {}".format(
        _DIFFICULTY_MAPPIN[difficulty], style
    )
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


def UI():
    with gr.Blocks() as demo:
        with gr.Tab("Generate Music by melody"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_path = gr.Dropdown(
                            choices=_AUDIOCRAFT_MODELS,
                            label="Select the model",
                            value="facebook/musicgen-melody-large",
                        )
                    with gr.Row():
                        duration = gr.Slider(
                            minimum=10,
                            maximum=60,
                            value=10,
                            label="Duration",
                            interactive=True,
                        )
                    with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(
                            label="Temperature", value=1.0, interactive=True
                        )
                        sample_rate = gr.Number(
                            label="output music sample rate",
                            value=32000,
                            interactive=True,
                        )
                        difficulty = gr.Radio(
                            ["Easy", "Medium", "Hard"],
                            label="Difficulty",
                            value="Easy",
                            interactive=True,
                        )
                        style = gr.Radio(
                            ["Jazz", "Classical Music", "Hip Hop"],
                            value="Classical Music",
                            label="music genre",
                            interactive=True,
                        )

                        def update_prompt(difficulty, style):
                            return gr.Textbox(
                            label="",
                            value=generate_prompt(difficulty, style),
                            interactive=False,
                            visible=False)
                        customize = gr.Checkbox(
                            label="Customize the prompt", interactive=True, value=False
                        )

                        _init_prompt = generate_prompt(difficulty.value, style.value)
                        prompt = gr.Textbox(
                            label="",
                            value=_init_prompt,
                            interactive=False,
                            visible=False,
                        )
                        customize.change(
                            fn=toggle_custom_prompt,
                            inputs=[customize, difficulty, style],
                            outputs=prompt,
                        )
                        difficulty.change(
                            update_prompt,
                            inputs=[difficulty, style],
                            outputs=prompt
                            )
                        style.change(
                            update_prompt,
                            inputs=[difficulty, style],
                            outputs=prompt
                            )
                        print(prompt)
                        with gr.Column():
                            optimize = gr.Button(
                                "Optimize the prompt", interactive=True
                            )
                        with gr.Column():
                            show_prompt = gr.Button("Show the prompt", interactive=True)
                            prompt_text = gr.Textbox(
                                "Optimized Prompt", interactive=False, visible=False
                            )
                        optimize.click(optimize_fn, inputs=[prompt], outputs=prompt)
                        show_prompt.click(
                            display_prompt, inputs=[prompt], outputs=prompt_text
                        )

                with gr.Column():
                    with gr.Row():
                        melody = gr.Audio(
                            sources=["microphone", "upload"],
                            label="Record or upload your audio",
                            # interactive=True,
                            show_label=True,
                        )
                    with gr.Row():
                        submit = gr.Button("Generate Music")
                        output_audio = gr.Audio(
                            "listen to the generated music", type="filepath"
                        )
                    with gr.Row():
                        transcribe_button = gr.Button("Transcribe")
                        d = gr.DownloadButton("Download the file", visible=False)
                        transcribe_button.click(
                            transcribe, inputs=[output_audio], outputs=d
                        )

            submit.click(
                fn=predict,
                inputs=[
                    model_path,
                    prompt,
                    melody,
                    duration,
                    topk,
                    topp,
                    temperature,
                    sample_rate,
                ],
                outputs=output_audio,
            )
            gr.Examples(
                examples=[
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/twinkle_twinkle_little_stars_mozart_20sec"
                            ".mp3",
                        ),
                        "Easy",
                        32000,
                        20,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/golden_hour_20sec.mp3",
                        ),
                        "Easy",
                        32000,
                        20,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/turkish_march_mozart_20sec.mp3",
                        ),
                        "Easy",
                        32000,
                        20,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/golden_hour_20sec.mp3",
                        ),
                        "Hard",
                        32000,
                        20,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/golden_hour_20sec.mp3",
                        ),
                        "Hard",
                        32000,
                        40,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/golden_hour_20sec.mp3",
                        ),
                        "Hard",
                        16000,
                        20,
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/audio/old_town_road20sec.mp3",
                            ),
                        "Hard",
                        32000,
                        40,
                        ],
                ],
                inputs=[melody, difficulty, sample_rate, duration],
                label="Audio Examples",
                outputs=[output_audio],
                # cache_examples=True,
            )

        with gr.Tab("Generate Music by image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image("Upload an image", type="filepath")
                    melody_condition = gr.Checkbox(
                        label="Generate music by melody", interactive=True, value=False
                    )
                    melody = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Record or upload your audio",
                        show_label=True,
                        visible=False,
                    )
                    melody_condition.change(
                        fn=toggle_melody_condition,
                        inputs=[melody_condition],
                        outputs=melody,
                    )
                    description = gr.Textbox(
                        label="Image Captioning",
                        show_label=True,
                        interactive=False,
                        visible=False,
                    )
                    prompt = gr.Textbox(
                        label="Generated Prompt",
                        show_label=True,
                        interactive=True,
                        visible=False,
                    )
                    show_prompt = gr.Checkbox(label="Show the prompt", interactive=True)
                    submit = gr.Button("submit", interactive=True, visible=True)
                    generate = gr.Button(
                        "Generate Music", interactive=True, visible=False
                    )

                with gr.Column():
                    with gr.Row():
                        model_path = gr.Dropdown(
                            choices=_AUDIOCRAFT_MODELS,
                            label="Select the model",
                            value="facebook/musicgen-large",
                        )
                    with gr.Row():
                        duration = gr.Slider(
                            minimum=10,
                            maximum=60,
                            value=10,
                            label="Duration",
                            interactive=True,
                        )
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(
                        label="Temperature", value=1.0, interactive=True
                    )
                    sample_rate = gr.Number(
                        label="output music sample rate", value=32000, interactive=True
                    )
                with gr.Column():
                    output_audio = gr.Audio(
                        "listen to the generated music",
                        type="filepath",
                        show_label=True,
                    )
                    transcribe_button = gr.Button("Transcribe")
                    d = gr.DownloadButton("Download the file", visible=False)
            submit.click(
                fn=post_submit,
                inputs=[show_prompt, model_path, image_input],
                outputs=[description, prompt, generate],
            )
            show_prompt.change(
                fn=show_caption,
                inputs=[show_prompt, description, prompt],
                outputs=[description, prompt, generate],
            )
            transcribe_button.click(transcribe, inputs=[output_audio], outputs=d)
            generate.click(
                fn=predict,
                inputs=[
                    model_path,
                    prompt,
                    melody,
                    duration,
                    topk,
                    topp,
                    temperature,
                    sample_rate,
                ],
                outputs=output_audio,
            )

            gr.Examples(
                examples=[
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/image/kids_drawing.jpeg",
                        ),
                        False,
                        None,
                        "facebook/musicgen-large",
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/image/cat.jpeg",
                        ),
                        False,
                        None,
                        "facebook/musicgen-large",
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/image/cat.jpeg",
                        ),
                        True,
                        "./data/audio/the_nutcracker_dance_of_the_reed_flutes.mp3",
                        "facebook/musicgen-melody-large",
                    ],
                    [
                        os.path.join(
                            os.path.dirname(__file__),
                            "./data/image/beach.jpeg",
                        ),
                        False,
                        None,
                        "facebook/audiogen-medium",
                    ],
                ],
                inputs=[image_input, melody_condition, melody, model_path],
                label="Audio Examples",
                outputs=[output_audio],
                # cache_examples=True,
            )

    demo.queue().launch()


if __name__ == "__main__":
    UI()
