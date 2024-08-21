import re

TEXT_TO_MUSIC_MODELS = [
    "facebook/musicgen-medium",
    "facebook/musicgen-small",
    "facebook/musicgen-large",
    'facebook/magnet-small-10secs', 
    'facebook/magnet-medium-10secs',
    'facebook/magnet-small-30secs', 
    'facebook/magnet-medium-30secs',
    "facebook/musicgen-stereo-small", 
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-large",
]

TEXT_TO_MIDI_MODELS = [
    "musiclang/musiclang-v2",
]

MELODY_CONTINUATION_MODELS = [
    "facebook/musicgen-medium",
    "facebook/musicgen-small",
    "facebook/musicgen-large",
]

TEXT_TO_SOUND_MODELS = [
    'facebook/audio-magnet-small', 
    'facebook/audio-magnet-medium',
    "facebook/audiogen-medium",
]

MELODY_CONDITIONED_MODELS = [
    "facebook/musicgen-melody",
    "facebook/musicgen-melody-large",
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-stereo-melody-large",
]

STEREO_MODEL = [
    "facebook/musicgen-stereo-small", 
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-stereo-melody-large",
]


MODEL_CARDS = {
    "text-to-music": TEXT_TO_MUSIC_MODELS,
    "text-to-midi": TEXT_TO_MIDI_MODELS,
    "text-to-sound": TEXT_TO_SOUND_MODELS,
    "melody-conditioned": MELODY_CONDITIONED_MODELS,
}

MODEL_DISCLAIMERS = {
    "facebook/musicgen-melody": "1.5B transformer decoder also supporting melody conditioning.",
    "facebook/musicgen-medium": "1.5B transformer decoder.",
    "facebook/musicgen-small": "300M transformer decoder.",
    "facebook/musicgen-large": "3.3B transformer decoder also supporting melody conditioning.",
    "facebook/musicgen-melody-large": "3.3B transformer decoder.",
    'facebook/magnet-small-10secs': "A 300M non-autoregressive transformer capable of generating 10-second music conditioned on text.",
    'facebook/magnet-medium-10secs': "A 1.5B parameters, 10 seconds music samples..",
    'facebook/magnet-small-30secs': "A 300M parameters, 30 seconds music samples.",
    'facebook/magnet-medium-30secs': "A 1.5B parameters, 30 seconds music samples.",
    # "musiclang/musiclang-v2": "This model generates music from text prompts.", TODO: Implement MusicLang
    'facebook/audio-magnet-small': "a 300M non-autoregressive transformer capable of generating 10 second sound effects conditioned on text.",
    'facebook/audio-magnet-medium': "10 second sound effect generation, 1.5B parameters.",
    "facebook/audiogen-medium": "1.5B transformer decoder capable of generating sound effects conditioned on text.",
}



def print_model_cards():
    for key, value in MODEL_CARDS.items():
        print(key, ":", value)
