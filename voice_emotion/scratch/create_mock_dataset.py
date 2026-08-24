import os
import numpy as np
import soundfile as sf
from pathlib import Path

def generate_mock_ravdess():
    base_dir = Path("c:/Users/Lenovo/Documents/hh/voice_emotion/data/raw/RAVDESS/Audio_Speech_Actors_01-24")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple tone
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 24 actors
    for actor in range(1, 25):
        actor_dir = base_dir / f"Actor_{actor:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        
        # 8 emotions
        for emotion in range(1, 9):
            # Filename pattern: 03-01-emotion-01-01-01-actor.wav
            filename = f"03-01-{emotion:02d}-01-01-01-{actor:02d}.wav"
            file_path = actor_dir / filename
            
            # Write dummy wav
            sf.write(str(file_path), y, sr)
            
    print(f"Generated mock RAVDESS dataset with 24 actors and 8 emotions at {base_dir}")

if __name__ == "__main__":
    generate_mock_ravdess()
