import subprocess
import sys

input_file = sys.argv[1]

output_file = "degraded_telephone.wav"

subprocess.run([
    "ffmpeg",
    "-y",
    "-i", input_file,
    "-ar", "8000",
    "-ac", "1",
    "-c:a", "pcm_mulaw",
    output_file
])

print("Telephone degradation completed!")
print("Output:", output_file)