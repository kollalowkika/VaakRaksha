import json
import numpy as np
import soundfile as sf
import torch

from models.AASIST import Model


AUDIO_FILE = "phone_test.wav"
MODEL_FILE = "models/weights/AASIST.pth"
CONFIG_FILE = "config/AASIST.conf"

DEVICE = torch.device("cpu")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 64600


print("Loading AASIST configuration...")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

model_config = config["model_config"]

print("Loading AASIST model...")

model = Model(model_config).to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

print("AASIST model loaded successfully")
print("Device:", DEVICE)


audio, sample_rate = sf.read(AUDIO_FILE)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

if sample_rate != SAMPLE_RATE:
    raise ValueError(
        f"Expected 16000 Hz audio, but got {sample_rate} Hz"
    )

print("Audio file:", AUDIO_FILE)
print("Audio duration:", round(len(audio) / SAMPLE_RATE, 2), "seconds")

print()
print("Starting streaming detection...")
print("-" * 55)


chunk_number = 1

for start in range(0, len(audio), CHUNK_SAMPLES):

    chunk = audio[start:start + CHUNK_SAMPLES]

    original_length = len(chunk)

    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(
            chunk,
            (0, CHUNK_SAMPLES - len(chunk))
        )

    waveform = torch.tensor(
        chunk,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(waveform)
        probabilities = torch.softmax(output[0], dim=1)

    ai_probability = probabilities[0][0].item()
    human_probability = probabilities[0][1].item()

    print(
        f"Chunk {chunk_number}: "
        f"AI/Spoof = {ai_probability * 100:.2f}% | "
        f"Human = {human_probability * 100:.2f}% | "
        f"Samples = {original_length}"
    )

    chunk_number += 1


print("-" * 55)
print("Streaming detection completed!")