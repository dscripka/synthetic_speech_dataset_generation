
# Custom interface for Nvidia Waveglow models

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from bentoml.utils import cloudpickle
from bentoml.exceptions import InvalidArgument
from bentoml.service.artifacts import BentoServiceArtifact

from flowtron import Flowtron

import torch

class WaveglowArtifact(BentoServiceArtifact):
    def __init__(self, name):
        super(WaveglowArtifact, self).__init__(name)
        self._model = None
        self.model_config = {
            "n_speakers": 2311,
            "n_speaker_dim": 128,
            "n_text": 185,
            "n_text_dim": 512,
            "n_flows": 2,
            "n_mel_channels": 80,
            "n_attn_channels": 640,
            "n_hidden": 1024,
            "n_lstm_layers": 2,
            "mel_encoder_n_hidden": 512,
            "n_components": 0,
            "mean_scale": 0.0,
            "fixed_gaussian": True,
            "dummy_speaker_embedding": False,
            "use_gate_layer": True
        }

    def pack(self, model, metadata=None):
        self._model = model
        return self

    def get(self):
        return self._model

    def save(self, dst):
        pass

    def load(self, path):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # load waveglow model
        waveglow = torch.load(os.path.join(path, 'waveglow_256channels_universal_v5.pt'))['model'].to(device)
        if device.type == "cuda":
            waveglow.cuda().half()
        waveglow.eval()
        
        # Load flowtron model
        flowtron = Flowtron(**self.model_config).to(device)
        state_dict = torch.load(os.path.join(path, "flowtron_libritts2p3k.pt"), map_location='cpu')['model'].state_dict()
        flowtron.load_state_dict(state_dict)
        _ = flowtron.eval()
        
        return self.pack({"waveglow": waveglow, "flowtron": flowtron})
