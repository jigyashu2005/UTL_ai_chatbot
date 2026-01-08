import torch
from transformers import pipeline
import os

# Define file path
AUDIO_FILENAME = r"C:\Users\HP\Downloads\WhatsApp Audio 2026-01-06 at 5.19.54 PM.mp4"

def main():
    # Setup FFMPEG
    import subprocess
    target_wav = "temp_audio.wav"
    
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Found ffmpeg at: {ffmpeg_exe}", flush=True)
        
        # Convert to WAV
        print("Converting MP4 to WAV...", flush=True)
        subprocess.run([ffmpeg_exe, "-y", "-i", AUDIO_FILENAME, "-ar", "16000", "-ac", "1", target_wav], check=True)
        
    except ImportError:
        print("imageio-ffmpeg not found. Trying to proceed without conversion...", flush=True)
    except Exception as e:
        print(f"Conversion Error: {e}", flush=True)

    print(f"Checking file: {target_wav}", flush=True)
    if not os.path.exists(target_wav):
        print("Error: WAV file creation failed.", flush=True)
        return

    print("Loading model (openai/whisper-tiny)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
    
    print("Transcribing...", flush=True)
    try:
        print("STEP 1: Load Audio", flush=True)
        # Use soundfile (simpler/robust)
        import soundfile as sf
        import numpy as np
        
        # soundfile.read returns (data, samplerate)
        # data is (samples, channels) e.g. (N, 2)
        raw_audio, sr = sf.read(target_wav)
        
        # If stereo, convert to mono
        if len(raw_audio.shape) > 1:
            raw_audio = raw_audio.mean(axis=1) # Average channels
            
        # Whisper expects float32
        raw_audio = raw_audio.astype(np.float32)
        
        input_data = {"raw": raw_audio, "sampling_rate": sr} 
        # Note: If sr != 16000, pipeline handles resampling usually, but we forced 16k in ffmpeg
        
        print(f"Audio Loaded: {raw_audio.shape}, SR={sr}", flush=True)
        
        print("STEP 2: Inference", flush=True)
        result = transcriber(input_data)
        
        print("STEP 3: Success", flush=True)
        print("\n--- TRANSCRIPTION RESULT ---\n", flush=True)
        print(result["text"], flush=True)
        
        out_path = os.path.join(os.getcwd(), "result.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
            
        print(f"Result written to {out_path}", flush=True)
        print("\n----------------------------", flush=True)
    except Exception as e:
        print(f"Error during transcription: {e}", flush=True)
        with open("error.log", "w") as f:
            f.write(str(e))
    
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
