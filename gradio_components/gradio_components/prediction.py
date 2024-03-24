import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import basic_pitch
import basic_pitch.inference
import gradio as gr
import torch
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import AudioGen, MusicGen
from basic_pitch import ICASSP_2022_MODEL_PATH
from transformers import AutoModelForSeq2SeqLM


def load_model(version="facebook/musicgen-melody"):
    if version in ["facebook/audiogen-medium"]:
        return AudioGen.get_pretrained(version)
    else:
        return MusicGen.get_pretrained(version)


def _do_predictions(
    model_file,
    model,
    texts,
    melodies,
    duration,
    progress=False,
    gradio_progress=None,
    target_sr=32000,
    target_ac=1,
    **gen_kwargs,
):
    print(
        "new batch",
        len(texts),
        texts,
        [None if m is None else (m[0], m[1].shape) for m in melodies],
    )
    be = time.time()
    processed_melodies = []
    model.set_generation_params(duration=duration)
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = (
                melody[0],
                torch.from_numpy(melody[1]).to(model.device).float().t(),
            )
            print(f"Input audio sample rate is {sr}")
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., : int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            # melody condition
            outputs = model.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=False,
            )
        else:
            if model_file == "facebook/audiogen-medium":
                # audio condition
                outputs = model.generate(texts, progress=progress)
            else:
                # text only
                outputs = model.generate(texts, progress=progress)

    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name,
                output,
                model.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            out_wavs.append(file.name)
    print("generation finished", len(texts), time.time() - be)
    return out_wavs


def predict(
    model_path,
    text,
    melody,
    duration,
    topk,
    topp,
    temperature,
    target_sr,
    progress=gr.Progress(),
):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    # if model_path:
    #     if not Path(model_path).exists():
    #         raise gr.Error(f"Model path {model_path} doesn't exist.")
    #     if not Path(model_path).is_dir():
    #         raise gr.Error(f"Model path {model_path} must be a folder containing "
    #                        "state_dict.bin and compression_state_dict_.bin.")
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    model = load_model(model_path)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    model.set_custom_progress_callback(_progress)

    wavs = _do_predictions(
        model_path,
        model,
        [text],
        [melody],
        duration,
        progress=True,
        target_ac=1,
        target_sr=target_sr,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        gradio_progress=progress,
    )
    return wavs[0]


def transcribe(audio_path):
    # model_output, midi_data, note_events = predict("generated_0.wav")
    model_output, midi_data, note_events = basic_pitch.inference.predict(
        audio_path=audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    with NamedTemporaryFile("wb", suffix=".mid", delete=False) as file:
        try:
            midi_data.write(file)
            print(f"midi file saved to {file.name}")
        except Exception as e:
            print(f"Error while writing midi file: {e}")
            raise e

    return gr.DownloadButton(
        value=file.name, label=f"Download MIDI file {file.name}", visible=True
    )
