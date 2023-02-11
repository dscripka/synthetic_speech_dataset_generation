# Readme

This repository contains text-to-speech (TTS) models and utilities designed produce synthetic training datasets for other speech-related models (e.g., [openWakeWord](https://github.com/dscripka/openWakeWord)).

It includes two specific open-source TTS models that I have found to be useful when generating synthetic speech. Specifically:

- [Nvidia Waveglow](https://github.com/NVIDIA/waveglow)
- [VITS](https://github.com/jaywalnut310/vits)

Note that the code in this repository varies greatly in quality and structure as it was derived from multiple sources. It is primarily meant for research and experimentation, and you are encouraged to makes changes and updates before relying on this code for production purposes. Also, these models are only trained on English TTS datasets (VCTK and LibriTTS), and will not produce accurate speech for other languages.

# Installation

First clone this repository:

```bash
git clone https://github.com/dscripka/synthetic_speech_dataset_generation
```

Then install the requirements into your virtual environment of choice:

```bash
pip install -r requirements.txt
```

If installing in an environment with GPUs available, you will need to update `requirements.txt` to include versions of Torch compatible with your GPU configuration. Note that while it is possible to generate data on CPUs only, the WAVEGLOW model will be very slow (e.g., 5-10 seconds per generation). The VITS model is somewhat faster on CPU (~1-3 seconds per generation), but for the large amounts of data generation that is often needed to train robust models, a GPU is *strongly* recommended.

The TTS models themselves are not stored in this repository and need to be downloaded separately. There is an included script that will download the files and place them in the appropriate location within the repository.

```bash
python download_tts_models.py
```

To test that everything is working correctly after these steps, use this command and listen to the output in the `generated_clips` directory that is created:

```bash
python generate_clips.py --model VITS --text "This is some test speech" --N 1 --output_dir generated_clips
```

# Usage

The primary way to generate synthetic speech is via the CLI in `generate_clips.py`. To see all of the possible arguments, use `python generate_clips.py --help`.

As a quick example of usage, the following command will generate 5000 clips of the phrase "turn on the office lights" using the Nvidia Waveglow model (on a GPU) trained on the LibriTTS dataset. Additionally, the `--max_per_speaker` argument will limit the number of generations for each of the ~2300 LibriTTS training voices to 1, and after that limit is reached a random voice will be created by [spherical interpolation](https://en.wikipedia.org/wiki/Slerp) of random speaker embeddings.

```
python generate_clips.py \
    --model WAVEGLOW \
    --enable_gpu \
    --text "turn on the office lights" \
    --N 5000 \
    --max_per_speaker 1 \
    --output_dir /path/to/output/directory
```

# License

The `generate_clips.py` code in this repository is licensed under Apache 2.0. The included TTS models (and the associated code from the source repos) have their own licenses, and you are strongly encouraged to review the original repositories to determine if the license is appropriate for a given use-case.

- [Nvidia Waveglow](https://github.com/NVIDIA/waveglow)
- [VITS](https://github.com/jaywalnut310/vits)