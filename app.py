import os
import shutil
import torch
import soundfile as sf
import numpy as np
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from models.AASIST import Model


app = FastAPI(title="VaakRaksha")


MODEL_FILE = "models/weights/AASIST.pth"
CONFIG_FILE = "config/AASIST.conf"

DEVICE = torch.device("cpu")


with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

model_config = config["model_config"]

model = Model(model_config).to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()


@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <html>

    <head>

        <title>VaakRaksha</title>

        <style>

            body {
                font-family: Arial;
                text-align: center;
                margin-top: 80px;
                background: #f4f7fb;
            }

            .box {
                background: white;
                width: 500px;
                margin: auto;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }

            h1 {
                font-size: 38px;
            }

            button {
                padding: 12px 25px;
                font-size: 16px;
                cursor: pointer;
            }

        </style>

    </head>


    <body>

        <div class="box">

            <h1>VaakRaksha</h1>

            <h2>Real-Time Voice-Clone Detection</h2>

            <p>
                Detect possible AI-generated voice and vishing audio.
            </p>


            <form
                action="/analyze"
                method="post"
                enctype="multipart/form-data"
            >

                <input
                    type="file"
                    name="audio"
                    accept=".wav"
                    required
                >

                <br><br>

                <button type="submit">
                    Analyze Voice
                </button>

            </form>

        </div>

    </body>

    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(audio: UploadFile = File(...)):

    file_name = "uploaded_audio.wav"


    with open(file_name, "wb") as buffer:

        shutil.copyfileobj(
            audio.file,
            buffer
        )


    waveform, sample_rate = sf.read(file_name)


    if waveform.ndim > 1:

        waveform = np.mean(
            waveform,
            axis=1
        )


    if sample_rate != 16000:

        os.remove(file_name)

        return """
        <html>

        <body style="text-align:center; margin-top:80px;">

            <h2>Unsupported Audio</h2>

            <p>
                Please upload a 16 kHz WAV audio file.
            </p>

            <a href="/">
                Go Back
            </a>

        </body>

        </html>
        """


    target_samples = 64600


    if len(waveform) < target_samples:

        waveform = np.pad(
            waveform,
            (0, target_samples - len(waveform))
        )

    else:

        waveform = waveform[:target_samples]


    input_tensor = torch.tensor(
        waveform,
        dtype=torch.float32
    ).unsqueeze(0)


    # AASIST inference

    with torch.no_grad():

        output = model(input_tensor)

        # AASIST returns:
        # (last_hidden, classification_logits)

        if isinstance(output, tuple):

            output = output[1]


        probabilities = torch.softmax(
            output,
            dim=1
        )


    ai_probability = (
        probabilities[0][0].item() * 100
    )

    human_probability = (
        probabilities[0][1].item() * 100
    )


    if ai_probability >= 50:

        result = "POSSIBLE AI / SPOOF VOICE"

        risk = "HIGH"

    else:

        result = "LIKELY HUMAN / BONAFIDE VOICE"

        risk = "LOW"


    os.remove(file_name)


    return f"""
    <html>

    <head>

        <title>VaakRaksha Result</title>

        <style>

            body {{
                font-family: Arial;
                text-align: center;
                margin-top: 60px;
                background: #f4f7fb;
            }}

            .result {{
                background: white;
                width: 550px;
                margin: auto;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}

            .score {{
                font-size: 22px;
                margin: 15px;
            }}

            .result-text {{
                font-size: 26px;
                font-weight: bold;
            }}

            .risk {{
                font-size: 24px;
                font-weight: bold;
            }}

        </style>

    </head>


    <body>

        <div class="result">

            <h1>VaakRaksha</h1>

            <h2>Detection Result</h2>


            <div class="score">

                AI / Spoof Probability:

                <b>{ai_probability:.2f}%</b>

            </div>


            <div class="score">

                Human Probability:

                <b>{human_probability:.2f}%</b>

            </div>


            <div class="score">

                Total Probability:

                <b>{ai_probability + human_probability:.2f}%</b>

            </div>


            <div class="risk">

                Risk Level:

                <b>{risk}</b>

            </div>


            <p class="result-text">

                {result}

            </p>


            <br>


            <a href="/">

                Analyze Another Audio

            </a>

        </div>

    </body>

    </html>
    """