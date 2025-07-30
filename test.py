import numpy as np
import requests
import uuid
import traceback
import librosa
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Config
SAMPLE_RATE = 16000
MAX_BYTES = 100000  # 1.5 MB approx
FLOAT32_SIZE = 4
REGION = "europe-west2"
ENDPOINT_ID = "8879581139128287232"
PROJECT = "avian-computer-467315-g3"
SERVICE_ACCOUNT_KEY_FILE = "key.json"

API_ROOT = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict"


def get_auth_headers():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


def load_audio_file(file_path):
    """Load audio file and resample to target sample rate"""
    try:
        print(f"Loading audio file: {file_path}")
        # Load audio file with librosa (handles various formats)
        audio, original_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"Original audio: {len(audio)} samples at {original_sr} Hz")

        # Resample if necessary
        if original_sr != SAMPLE_RATE:
            print(f"Resampling from {original_sr} Hz to {SAMPLE_RATE} Hz")
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=SAMPLE_RATE)

        print(f"Final audio: {len(audio)} samples at {SAMPLE_RATE} Hz")
        return audio.astype(np.float32)

    except Exception as e:
        print(f"Error loading audio file: {e}")
        traceback.print_exc()
        return None


def prepare_audio_chunk(audio):
    max_samples = MAX_BYTES // FLOAT32_SIZE
    if len(audio) > max_samples:
        print(f"Audio too long, truncating from {len(audio)} to {max_samples} samples")
        audio = audio[:max_samples]
    else:
        print(f"Audio length {len(audio)} samples fits in max size {max_samples}")
    return audio


def send_audio(audio):
    headers = get_auth_headers()
    chunk_tensor = audio.astype(np.float32).tolist()

    payload = {
        "id": str(uuid.uuid4()),
        "inputs": [
            {
                "name": "audio",
                "shape": [len(chunk_tensor)],
                "datatype": "FP32",
                "parameters": {},
                "data": chunk_tensor,
            },
            {
                "name": "sampling_rate",
                "shape": [1],
                "datatype": "INT32",
                "parameters": {},
                "data": [SAMPLE_RATE],
            },
        ],
    }

    try:
        response = requests.post(API_ROOT, headers=headers, json=payload)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:", response.json())
        else:
            print("Error response:", response.text)
    except Exception:
        print("Exception during request:")
        traceback.print_exc()


if __name__ == "__main__":
    # Specify your WAV file path here
    wav_file_path = "test.wav"  # Change this to your actual file path

    audio_data = load_audio_file(wav_file_path)
    if audio_data is not None:
        audio_chunk = prepare_audio_chunk(audio_data)
        send_audio(audio_chunk)
    else:
        print("Failed to load audio file")