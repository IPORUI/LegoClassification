import os, sys
import subprocess
from pathlib import Path

BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 3.0\\"

if __name__ == '__main__':
    path = Path("generate_dataset.py").resolve()
    subprocess.run(BLENDER_PATH + "blender --background -noaudio --threads 0 --verbose 0 --python " + str(path))