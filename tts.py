import torchaudio as ta
import torch
from huggingface_hub import snapshot_download
from chatterbox.tts_turbo import ChatterboxTurboTTS, REPO_ID

# Download model snapshot explicitly WITHOUT sending an auth token
# -> token=False prevents sending Authorization header
local_path = snapshot_download(
    repo_id=REPO_ID,
    token=False,  # disable auth token for this download
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
)

# Load the Turbo model from the downloaded local path
model = ChatterboxTurboTTS.from_local(local_path, device='cpu')

# Generate with Paralinguistic Tags
text = "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?"

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")

ta.save("test-turbo.wav", wav, model.sr)