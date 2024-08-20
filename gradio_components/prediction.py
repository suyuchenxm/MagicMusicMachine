import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import basic_pitch
import basic_pitch.inference
import gradio as gr
import torch
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
# from audiocraft.models import AudioGen, MusicGen
from basic_pitch import ICASSP_2022_MODEL_PATH
# from transformers import AutoModelForSeq2SeqLM
from gradio_components.model_cards import load_model
from concurrent.futures import ProcessPoolExecutor
import warnings


pool = ProcessPoolExecutor(4)
class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
                
file_cleaner = FileCleaner()

def inference_musicgen_text_to_music(model, configs, text, num_outputs=1):
    model.set_generation_params(
        **configs
    )
    descriptions = [text for _ in range(num_outputs)]
    output = model.generate(descriptions=descriptions ,progress=True, return_tokens=True)
    return output

def inference_musicgen_continuation(model, configs, text, prompt_waveform, prompt_sr, num_outputs=1):
    model.set_generation_params(
        **configs
    )
    descriptions = [text for _ in range(num_outputs)]
    prompt = [prompt_waveform for _ in range(num_outputs)]
    output = model.generate_continuation(prompt, prompt_sample_rate=prompt_sr, descriptions=descriptions, progress=True, return_tokens=True)
    return output

def inference_musicgen_melody_condition(model, configs, text, prompt_waveform, prompt_sr, num_outputs=1):
    model.set_generation_params(**configs)
    melody_waveform = [prompt_waveform for _ in range(num_outputs)]
    descriptions = [text for _ in range(num_outputs)]
    output = model.generate_with_chroma(
        descriptions=descriptions,
        melody_wavs=melody_waveform,
        melody_sample_rate=prompt_sr,
        progress=True, 
        return_tokens=True
    )
    return output

def inference_magnet(model, configs, text, num_outputs=1):
    model.set_generation_params(
        **configs
    )
    descriptions = [text for _ in range(num_outputs)]
    output = model.generate(descriptions=descriptions, progress=True, return_tokens=True)
    return output

def inference_magnet_audio(model, configs, text, num_outputs=1):
    model.set_generation_params(
        **configs
    )
    descriptions = [text for _ in range(num_outputs)]
    output = model.generate(descriptions=descriptions, progress=True, return_tokens=True)
    return output
    
def inference_audiogen(model, configs, text, num_outputs=1):
    model.set_generation_params(
        **configs
    )
    descriptions = [text for _ in range(num_outputs)]
    output = model.generate(descriptions=descriptions, progress=True, return_tokens=True)
    return output

def inference_musiclang():
    # TODO: Implement MusicLang
    pass


def process_audio(gr_audio, model):
    audio, sr = torch.from_numpy(gr_audio[1]).to(model.device).float().t(), gr_audio[0]
    return audio, sr

_MODEL_INFERENCES = {
    "facebook/musicgen-melody": inference_musicgen_melody_condition,
    "facebook/musicgen-medium": inference_musicgen_text_to_music,
    "facebook/musicgen-small": inference_musicgen_text_to_music,
    "facebook/musicgen-large": inference_musicgen_text_to_music,
    "facebook/musicgen-melody-large": inference_musicgen_melody_condition,
    "facebook/magnet-small-10secs": inference_magnet,
    "facebook/magnet-medium-10secs": inference_magnet,
    "facebook/magnet-small-30secs": inference_magnet,
    "facebook/magnet-medium-30secs": inference_magnet,
    "facebook/audio-magnet-small": inference_magnet_audio,
    "facebook/audio-magnet-medium": inference_magnet_audio,
    "facebook/audiogen-medium": inference_audiogen,
    "musicgen-continuation": inference_musicgen_continuation,
}

def _do_predictions(
    model_file,
    model,
    text,
    melody = None,
    mel_sample_rate=None,
    progress=False,
    num_generations=1,
    **gen_kwargs,
):
    print(
        "new generation",
        text,
        None if melody is None else melody.shape
    )
    be = time.time()
    try:
        if melody is not None:
            # melody condition or continuation
            inderence_func = _MODEL_INFERENCES[model_file]
            outputs = inderence_func(model, gen_kwargs, text, melody, mel_sample_rate, num_generations)
        else:
            # text-to-music, text-to-sound
            inderence_func = _MODEL_INFERENCES[model_file]
            outputs = inderence_func(model, gen_kwargs, text, num_generations)

    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    out_audios = []
    video_processes = []
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
            video_processes.append(pool.submit(make_waveform, file.name))
            out_audios.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [video.result() for video in video_processes]
    for video in out_videos:
        file_cleaner.add(video)
    
    print("generation finished", len(outputs), time.time() - be)
    return out_audios, out_videos

def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out
    
def predict(
    model,
    model_version,
    generation_configs,
    prompt_text=None,
    prompt_wav=None,
    progress=gr.Progress(),
    num_generations=1,
):
    global INTERRUPTING
    INTERRUPTING = False
    # progress(0, desc="Loading model...")
    
    max_generated = 0

    melody, mel_sample_rate = process_audio(prompt_wav) if prompt_wav is not None else None

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    model.set_custom_progress_callback(_progress)

    audios, waveforms = _do_predictions(
        model_version,
        model,
        prompt_text,
        melody,
        mel_sample_rate,
        progress=True,
        num_generations = num_generations,
        **generation_configs,
    )
    return audios, waveforms


def transcribe(audio_path):
    """
    Transcribe an audio file to MIDI using the basic_pitch model.
    """
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


