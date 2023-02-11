# Imports and arguments

from pathlib import Path
import argparse
import re
import os
import random
import numpy as np

parser = argparse.ArgumentParser("Generates 16khz, 16-bit PCM, single channel synthetic speech to serve as training data for wakeword detection systems")
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("--model", metavar='', type=str, help="Which model to use for generated speech (VITS or WAVEGLOW)")
requiredNamed.add_argument("--enable_gpu", metavar='', type=bool, default=False, help="""Whether to use a GPU (if available) for generation.
                                                                                 This is reccomended as the speed of generation on CPU will be
                                                                                 significantly slower compared to GPU execution.""")
requiredNamed.add_argument("--text", metavar='', type=str, help="The text to generate")
requiredNamed.add_argument('--input_file', metavar='', type=str, help="""A text file where each line is the text of one clip to generate.
                           if this argument is provided, the `text` argument is ignored.""", default="")
requiredNamed.add_argument("--N", metavar='', type=int, default=1,
                           help="""How many total phrases to generate. Note that sometimes generation fails,
                                so the total number of valid saved clips may be < N.""")
requiredNamed.add_argument("--output_dir", metavar='', type=str, help="The target directory for the generated clips")
requiredNamed.add_argument("--max_per_speaker", metavar='', type=int, default=1,
                           help="""How many times to generate text with a give speaker. If N is greater than the total number of speakers 
                          random speaker embeddings (i.e., random mixtures of voices) will be used for all remaining generations. (Default: 1)""")
requiredNamed.add_argument("--truncate_at_pause", metavar='', type=bool, default=False,
                           help="""Whether to truncate the audio after the first pause as identified from silence detection heuristics.
                           Useful when you only want the first word or two from a generated clip, but the pronunciation isn't correct without
                           a longer text input (a common problem since most TTS models are trained on longer phrases).
                           Note that the VITS model will generally pause after a ',', and the WAVEGLOW model pauses after a '-'.
                           Because the silence detection hueristics are only approximate, in practice enabling this functionality 
                           will reduce the total number of requested generations (N) by 0-10 percent.""")
requiredNamed.add_argument("--pause_min_start", metavar='', type=int, default=8000,
                           help="""The minimum number of samples (@16khz) to look for a pause, if `truncate_at_pause` is set to True.
                           Pause before this time will be ignored.""")
requiredNamed.add_argument("--pause_max_end", metavar='', type=int, default=16000,
                           help="""The maximum number of samples (@16khz) to look for a pause, if `truncate_at_pause` is set to True.
                           Pauses after this time will be ignored.""")
requiredNamed.add_argument("--speaking_speed", metavar='', type=float, default=1.0,
                           help="""The speaking speed of the generation. For the VITS model, this is a configurable parameter.
                           for the WAVEGLOW model, the clips is adjusted after generation with librisa.effects.time_stretch.""")

# Helper functions
def get_silence_times(dat, max_end, min_start=0, std_thresh=100, longest_first = True):
    if dat.max() == 0:
        return []
        
    x = np.arange(min_start, len(dat), 100)
    y = np.array([np.std(dat[i:i+200]) for i in range(min_start, len(dat), 100)])
    silence_spans = ''.join([str(int(i)) for i in (y>0) & (y <= std_thresh)])
    std_thresh = np.percentile(y, 30)
    silence_spans = [(x[i.span()[0]], x[i.span()[1]-1]) for i in list(re.finditer("1+", silence_spans))]
    silence_spans = [i for i in silence_spans if i[0] <= max_end]
    silence_spans = [i for i in silence_spans if i[1] - i[0] > 100]
    if longest_first:
        silence_spans = sorted(silence_spans, key=lambda x: x[1] - x[0], reverse=True)
    return silence_spans

def generate_speaker_ids(N, n_speakers, n_mix=2):
    ids_original = [[i] for i in np.arange(0, n_speakers)]
    if N < n_speakers:
        ids = ids_original[0:N]
    elif N > n_speakers:
        ids_original = (ids_original*args.max_per_speaker)[0:N]
        ids_random = [np.random.randint(0,n_speakers,n_mix).tolist() for _ in range(N - len(ids_original))]
        ids = ids_original + ids_random
    return ids

# Build sentence variations
def get_random_variation(txt):
    # Get possible words
    positions = txt.split()
    words = [i.split("|") for i in positions]

    # Build variations
    variation = " ".join([random.choice(i) for i in words])
    return re.sub("\s+", " ", variation)

# Get random words
en_words = open(os.path.join("data", "20k.txt"), "r").readlines()
en_words = [i.strip() for i in en_words][0:10000]

# Parse arguments
args = parser.parse_args()

# Create output directory if it doesn't exist
out_dir = Path(args.output_dir)
if not out_dir.exists():
    os.mkdir(out_dir)

if args.model == "VITS":
    # Imports for VITS model
    import sys
    import os
    sys.path.append(os.path.join(Path(__file__).parent.absolute(), "models", "vits"))

    if args.enable_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import torchaudio
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    import numpy as np
    import librosa
    import uuid
    from tqdm import tqdm

    import commons
    import utils
    from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
    from models import SynthesizerTrn
    from text.symbols import symbols
    from text import text_to_sequence
    from phonemizer import phonemize
    from phonemizer.phonemize import _phonemize
    from phonemizer.separator import default_separator, Separator
    from phonemizer.backend import EspeakBackend

    from scipy.io.wavfile import write

    # Helper functions/classes for VITs model
    class VitsModel:
        def __init__(self, hparams_path, checkpoint_path, sample_rate=16000, cuda=False):
            self.cuda_flag = cuda
            self.hps = utils.get_hparams_from_file(hparams_path)
            self.model = self.load_model(hparams_path, checkpoint_path, cuda)
            self.model.eval()
            self.resampler = torchaudio.transforms.Resample(self.hps.data.sampling_rate, sample_rate)

        def get_text(self, text):
            text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
            if self.hps.data.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)
            return text_norm

        def generate_speech(self, txt, speaker_id=0, noise_bounds=(0.667, 0.667), duration_bounds=(1.0, 1.0)):
            stn_tst = self.get_text(txt)
            noise_scale = np.random.uniform(noise_bounds[0], noise_bounds[1])
            duration_scale = np.random.uniform(duration_bounds[0], duration_bounds[1])
            with torch.no_grad():
                try:
                    if self.cuda_flag:
                        x_tst = stn_tst.cuda().unsqueeze(0)
                        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                        sid = torch.LongTensor(speaker_id).cuda()
                        audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8, length_scale=duration_scale)[0][0,0].data.cpu().float()
                    else:
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                        sid = torch.LongTensor(speaker_id)
                        audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8, length_scale=duration_scale)[0][0,0].data.float()

                    audio = self.resampler(audio)  #resample to 16khz
                    audio = (audio*32767).numpy().astype(np.int16)  #convert to 16-bit PCM format
                except AssertionError:
                    audio = None

            return audio

        def load_model(self, hparams_path, checkpoint_path, cuda):
            net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers = self.hps.data.n_speakers,
                **self.hps.model)
            if cuda:
                net_g.cuda()

            _ = net_g.eval()

            _ = utils.load_checkpoint(checkpoint_path, net_g, None)

            return net_g

    # Load VITS model
    print("Loading VITS model...")
    model = VitsModel(
        hparams_path=os.path.join(Path(__file__).parent.absolute(), "models/vits/configs/vctk_base.json"),
        checkpoint_path=os.path.join(Path(__file__).parent.absolute(), "models/vits/pretrained_models/pretrained_vctk.pth"),
        cuda=False
    )

    # Get speaker ids and text for each generation
    if args.input_file == "":
        ids = generate_speaker_ids(args.N, n_speakers=109, n_mix=5)  #for VITS model trained on VCTK dataset
        texts = []
        for i in range(len(ids)):
            text = re.sub("<any>", lambda x: random.choice(en_words), args.text)
            if "|" in text:
                text = get_random_variation(text)
            texts.append(text)
    else:
        with open(args.input_file, 'r') as f:
            texts = f.readlines()
            texts = [i.strip() for i in texts]
        ids = generate_speaker_ids(len(texts), n_speakers=109, n_mix=5)  #for VITS model trained on VCTK dataset

    # Generate audio
    sr = 16000

    cnt = 0
    for i, text in tqdm(zip(ids, texts), total=len(ids), desc="Generating clips"):
        audio = model.generate_speech(txt=text, speaker_id=i, noise_bounds=(0.667, 1.5),
                                      duration_bounds=(0.8, 1.2) if args.speaking_speed == 1.0 else (args.speaking_speed, args.speaking_speed))
        if args.truncate_at_pause:
            breaks = get_silence_times(audio, min_start = args.pause_min_start, max_end=args.pause_max_end)
            if breaks and audio is not None:
                audio = audio[0:breaks[0][0]]
            else:  # try once more if initial generation fails
                audio = model.generate_speech(txt=text, speaker_id=i, noise_bounds=(0.667, 1.5),
                                              duration_bounds=(0.8, 1.2) if args.speaking_speed == 1.0 else (args.speaking_speed, args.speaking_speed))
                breaks = get_silence_times(audio, min_start = args.pause_min_start, max_end=args.pause_max_end)
                if breaks and audio is not None:
                    audio = audio[0:breaks[0][0]]
                else:
                    continue
        
        # Save clips
        if audio is not None:
            write(os.path.join(args.output_dir, uuid.uuid4().hex + ".wav"), sr, audio)
            cnt += 1

    print(f"{cnt} clips generated!")

elif args.model == "WAVEGLOW":
    # Imports for waveglow model
    import os

    if args.enable_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import bentoml
    import numpy as np
    import base64
    from scipy.io.wavfile import write
    from tqdm import tqdm
    import librosa
    import uuid
    import scipy.io.wavfile
    from pathlib import Path
    import torch

    # Load model
    print("Loading WAVEGLOW model...")
    model = bentoml.load_from_dir(os.path.join(Path(__file__).parent.absolute(), "models/waveglow"))

    # Get speaker ids and text for each generation
    if args.input_file == "":
        ids = generate_speaker_ids(args.N, n_speakers=2311, n_mix=2) # for waveglow model trained on LibriTTS dataset
        texts = []
        for i in range(len(ids)):
            text = re.sub("<any>", lambda x: random.choice(en_words), args.text)
            if "|" in text:
                text = get_random_variation(text)
            texts.append(text)
    else:
        with open(args.input_file, 'r') as f:
            texts = f.readlines()
            texts = [i.strip() for i in texts]
        ids = generate_speaker_ids(len(texts), n_speakers=2311, n_mix=2) # for waveglow model trained on LibriTTS dataset


    # Generate audio
    sr = 16000
    def generate(txt, speaker_id):
        input_json = {
            "text": txt,
            "speaker_id": speaker_id, #max of 2310
            "sigma": 0.8, # controls variability in the output speech
            "n_frames": 300,
            "sample_rate": sr
        }

        clip = model.generate(input_json)
        clip = np.frombuffer(base64.b64decode(clip), dtype=np.int16)
        return clip

    cnt = 0
    for i, text in tqdm(zip(ids, texts), total=len(ids), desc="Generating clips"):
        audio = generate(txt=text, speaker_id=i)

        if args.truncate_at_pause:
            breaks = get_silence_times(audio, min_start = args.pause_min_start, max_end=args.pause_max_end)
            if breaks:
                audio = audio[0:breaks[0][0]]
            else:  # try once more if initial generation fails
                audio = generate(txt=text, speaker_id=i)
                breaks = get_silence_times(audio, min_start = args.pause_min_start, max_end=args.pause_max_end)
                if breaks:
                    audio = audio[0:breaks[0][0]]
                else:
                    continue
        
        # Adjust clip speed, as needed
        if args.speaking_speed != 1.0:
            audio = (librosa.effects.time_stretch(audio/32767.0, rate=args.speaking_speed)*32767).astype(np.int16)

        # Save clips
        write(os.path.join(args.output_dir, uuid.uuid4().hex + ".wav"), sr, audio)
        cnt += 1

    print(f"{cnt} clips generated!")
