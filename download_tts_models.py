# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import functools
import pathlib
import shutil
import requests
import os
from tqdm.auto import tqdm

# Helper function to download files (from https://stackoverflow.com/a/63831344)
def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

# Download files
print("Downloading TTS models...\n")
vits_model = "https://f002.backblazeb2.com/file/openwakeword-resources/tts_models/pretrained_vctk.pth"
waveglow_model = "https://f002.backblazeb2.com/file/openwakeword-resources/tts_models/waveglow_256channels_universal_v5.pt"
flowtron_libritts_model = "https://f002.backblazeb2.com/file/openwakeword-resources/tts_models/flowtron_libritts2p3k.pt"

download(vits_model, vits_model.split("/")[-1])
download(waveglow_model, waveglow_model.split("/")[-1])
download(flowtron_libritts_model, flowtron_libritts_model.split("/")[-1])

# Move model files to correct locations
print("\nMoving model files.....")
shutil.move(vits_model.split("/")[-1], os.path.join("models", "vits", "pretrained_models", vits_model.split("/")[-1]))
shutil.move(waveglow_model.split("/")[-1], os.path.join("models", "waveglow", "TextToSpeechModel", "artifacts", waveglow_model.split("/")[-1]))
shutil.move(flowtron_libritts_model.split("/")[-1], os.path.join("models", "waveglow", "TextToSpeechModel", "artifacts", flowtron_libritts_model.split("/")[-1]))
print("Done!")