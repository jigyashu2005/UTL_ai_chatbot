import os
import torch
import torchaudio
import sys

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)

wav_file = "temp_audio.wav"
if not os.path.exists(wav_file):
    print("WAV missing!")
    sys.exit(1)

print("Attempting to load WAV with torchaudio...")
try:
    wav, sr = torchaudio.load(wav_file)
    print(f"Success! WAV Shape: {wav.shape}, SR: {sr}")
except Exception as e:
    print(f"Torchaudio Error: {e}")
    # Try soundfile directly
    import soundfile as sf
    try:
        data, samplerate = sf.read(wav_file)
        print(f"Soundfile read success: {data.shape}, {samplerate}")
    except Exception as ex:
        print(f"Soundfile Error: {ex}")

print("Loading Transformers pipeline...")
try:
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    print("Pipeline loaded.")
    
    print("Running inference...")
    res = pipe(wav_file)
    print("Result:", res)
    
except Exception as e:
    print(f"Pipeline Error: {e}")
