
# Create Bento for text to speech

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.pytorch import PytorchModelArtifact

from waveglow_artifact import WaveglowArtifact
from glow import WaveGlow
from data import Data

import re
import numpy as np
import base64
import pathlib
import scipy

@env(
    pip_packages=[
        "bentoml==0.12.1",
        "torch==1.7.1",
        "numpy==1.19.2",
        "inflect==4.1.0",
        "scipy==1.5.2",
        "Unidecode==1.0.22",
        "librosa==0.6.0"
    ]
)
@artifacts([WaveglowArtifact('model')])
class TextToSpeechModel(BentoService):
    """
    A model that converts text into spoken speech
    """
    def __init__(self):
        super(TextToSpeechModel, self).__init__()
        self.data_config = {
            "text_cleaners": ["flowtron_cleaners"],
            "p_arpabet": 0.5,
            "cmudict_path": str(pathlib.Path(__file__).parent.absolute()) + "/artifacts/cmudict_dictionary",
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "max_wav_value": 32768.0
        }
        training_files = str(pathlib.Path(__file__).parent.absolute()) + \
            "/artifacts/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt"
        self.tokenizer = Data(training_files, **self.data_config)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    @api(input=JsonInput())
    def generate(self, parsed_json, speaker_id=[24], sample_rate=22050, sigma=0.8, n_frames=300):
        text = parsed_json['text']
        if parsed_json.get('speaker_id', None):
            speaker_id = parsed_json['speaker_id']
        if parsed_json.get('sample_rate', None):
            sample_rate = parsed_json['sample_rate']
        if parsed_json.get('sigma', None):
            sigma = parsed_json['sigma']
        if parsed_json.get('n_frames', None):
            n_frames = parsed_json['n_frames']

        sentences = re.split('\,|\.|\;|\?|\!', text) # tokenize into chunks by punctuation
        sentences = [i for i in sentences if i != ""]
        audio = []
        for sentence in sentences:
            speaker_vecs = torch.tensor(speaker_id)[None].to(self.device)#.cuda()
            text = self.tokenizer.get_text(sentence)[None].to(self.device)#.cuda()

            with torch.no_grad():
                # residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
                residual = (torch.FloatTensor(1, 80, n_frames).normal_() * sigma).to(self.device)
                mels, attentions = self.artifacts.model.get("flowtron").infer(residual, speaker_vecs, text)

            if self.device.type == "cuda":
                mels = mels.half()

            clip = self.artifacts.model.get("waveglow").infer(mels, sigma=sigma).float()
            clip = clip.cpu().numpy()[0]
            clip = clip / np.abs(clip).max() # normalize audio
            if sample_rate != 22050:
                audio.append(scipy.signal.resample(clip, int(sample_rate*len(clip)/22050))) #convert to desired khz for playback
            else:
                audio.append(clip)
        
        speech = np.concatenate((audio))
        speech = (speech*32767).astype(np.int16) # convert to 16-bit PCM data
        return base64.b64encode(speech.tobytes()).decode('utf-8')