# src/orchestrator/main.py
from __future__ import annotations

import os
import sys
import signal
from dataclasses import dataclass
from typing import Optional

# Make sure src/ is importable if you run from repo root
# (You can remove this if your project is packaged properly)
sys.path.append("src")


# -------------------------
# Minimal .env loader
# -------------------------
def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


# -------------------------
# Config
# -------------------------
@dataclass
class OrchestratorConfig:
    enable_stt: bool = True
    enable_tts: bool = True
    text_input_fallback: bool = True

    print_transcripts: bool = True
    print_llm_output: bool = True

    # LLM env
    llm_model_path: str = ""
    llm_context_tokens: int = 2048
    llm_max_tokens: int = 150
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_threads: int = 8
    llm_gpu_layers: int = 0

    max_turns: int = 6
    max_chars_per_msg: int = 500
    max_response_chars: int = 400

    # TTS env (Piper)
    tts_model_path: str = ""
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_noise_w_scale: float = 0.8
    tts_volume: float = 1.0
    tts_output_device: Optional[int] = None

    # STT env
    stt_model: str = "tiny"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_language: str = "en"
    stt_rolling_window_sec: float = 8.0
    stt_infer_interval_ms: int = 400
    stt_emit_partials: bool = True
    stt_partial_min_delta: int = 5
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_frame_ms: int = 20
    
    # VAD settings
    vad_aggressiveness: int = 2
    vad_energy_threshold: float = 0.0
    vad_smooth_window: int = 5
    vad_silence_ms: int = 800
    
    # Segmenter settings
    start_trigger_frames: int = 2
    min_utterance_ms: int = 200
    max_utterance_ms: int = 10_000
    
    # Debug logging
    debug_save_conversations: bool = False
    debug_save_audio: bool = False
    debug_log_dir: str = "logs/conversations"
    debug_show_pipeline_status: bool = False
    debug_show_vad_state: bool = False


def load_config() -> OrchestratorConfig:
    load_dotenv(".env")

    cfg = OrchestratorConfig(
        enable_stt=env_bool("ENABLE_STT", True),
        enable_tts=env_bool("ENABLE_TTS", True),
        text_input_fallback=env_bool("TEXT_INPUT_FALLBACK", True),

        print_transcripts=env_bool("PRINT_TRANSCRIPTS", True),
        print_llm_output=env_bool("PRINT_LLM_OUTPUT", True),

        llm_model_path=env_str("LLM_MODEL_PATH", ""),
        llm_context_tokens=env_int("LLM_CONTEXT_TOKENS", 2048),
        llm_max_tokens=env_int("LLM_MAX_TOKENS", 150),
        llm_temperature=env_float("LLM_TEMPERATURE", 0.7),
        llm_top_p=env_float("LLM_TOP_P", 0.9),
        llm_repeat_penalty=env_float("LLM_REPEAT_PENALTY", 1.1),
        llm_threads=env_int("LLM_THREADS", 8),
        llm_gpu_layers=env_int("LLM_GPU_LAYERS", 0),

        max_turns=env_int("LLM_MAX_TURNS", 6),
        max_chars_per_msg=env_int("LLM_MAX_CHARS_PER_MSG", 500),
        max_response_chars=env_int("LLM_MAX_RESPONSE_CHARS", 400),

        tts_model_path=env_str("TTS_MODEL_PATH", ""),
        tts_length_scale=env_float("TTS_LENGTH_SCALE", 1.0),
        tts_noise_scale=env_float("TTS_NOISE_SCALE", 0.667),
        tts_noise_w_scale=env_float("TTS_NOISE_W_SCALE", 0.8),
        tts_volume=env_float("TTS_VOLUME", 1.0),

        stt_model=env_str("STT_MODEL", "tiny"),
        stt_device=env_str("STT_DEVICE", "cpu"),
        stt_compute_type=env_str("STT_COMPUTE_TYPE", "int8"),
        stt_beam_size=env_int("STT_BEAM_SIZE", 1),
        stt_language=env_str("STT_LANGUAGE", "en"),
        stt_rolling_window_sec=env_float("STT_ROLLING_WINDOW_SEC", 8.0),
        stt_infer_interval_ms=env_int("STT_INFER_INTERVAL_MS", 400),
        stt_emit_partials=env_bool("STT_EMIT_PARTIALS", True),
        stt_partial_min_delta=env_int("STT_PARTIAL_MIN_DELTA", 5),
        
        audio_sample_rate=env_int("AUDIO_SAMPLE_RATE", 16000),
        audio_channels=env_int("AUDIO_CHANNELS", 1),
        audio_frame_ms=env_int("AUDIO_FRAME_MS", 20),
        
        vad_aggressiveness=env_int("VAD_AGGRESSIVENESS", 2),
        vad_energy_threshold=env_float("VAD_ENERGY_THRESHOLD", 0.0),
        vad_smooth_window=env_int("VAD_SMOOTH_WINDOW", 5),
        vad_silence_ms=env_int("VAD_SILENCE_MS", 800),
        
        start_trigger_frames=env_int("START_TRIGGER_FRAMES", 2),
        min_utterance_ms=env_int("MIN_UTTERANCE_MS", 200),
        max_utterance_ms=env_int("MAX_UTTERANCE_MS", 10_000),
        
        debug_save_conversations=env_bool("DEBUG_SAVE_CONVERSATIONS", False),
        debug_save_audio=env_bool("DEBUG_SAVE_AUDIO", False),
        debug_log_dir=env_str("DEBUG_LOG_DIR", "logs/conversations"),
        debug_show_pipeline_status=env_bool("DEBUG_SHOW_PIPELINE_STATUS", False),
        debug_show_vad_state=env_bool("DEBUG_SHOW_VAD_STATE", False),
    )

    dev = env_str("TTS_OUTPUT_DEVICE", "").strip()
    if dev:
        try:
            cfg.tts_output_device = int(dev)
        except ValueError:
            cfg.tts_output_device = None

    return cfg


# -------------------------
# Orchestrator
# -------------------------
class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self._running = False

        # ---- Optional: Conversation Logger
        self.logger = None
        if cfg.debug_save_conversations:
            from utils.conversation_logger import ConversationLogger
            self.logger = ConversationLogger(
                log_dir=cfg.debug_log_dir,
                save_audio=cfg.debug_save_audio,
                save_text=True,
                sample_rate=cfg.audio_sample_rate
            )
            print(f"[DEBUG] Conversation logging enabled -> {cfg.debug_log_dir}")

        # ---- LLM core (always required)
        from llm.llm_engine import LLMEngine, LLMConfig
        from llm.memory import ConversationMemory, MemoryConfig
        from llm.prompt_manager import PromptManager, PromptConfig

        if not cfg.llm_model_path:
            raise ValueError("LLM_MODEL_PATH is required in .env")

        self.llm = LLMEngine(
            LLMConfig(
                model_path=cfg.llm_model_path,
                context_tokens=cfg.llm_context_tokens,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
                top_p=cfg.llm_top_p,
                repeat_penalty=cfg.llm_repeat_penalty,
                threads=cfg.llm_threads,
                gpu_layers=cfg.llm_gpu_layers,
            )
        )

        self.memory = ConversationMemory(
            MemoryConfig(max_turns=cfg.max_turns, max_chars_per_msg=cfg.max_chars_per_msg)
        )

        self.prompt_mgr = PromptManager(
            PromptConfig(max_response_chars=cfg.max_response_chars)
        )

        # ---- Optional: TTS
        self.tts = None
        if cfg.enable_tts:
            if not cfg.tts_model_path:
                raise ValueError("ENABLE_TTS=1 but TTS_MODEL_PATH is missing")
            from tts.tts_engine import PiperTTSEngine, PiperTTSConfig
            self.tts = PiperTTSEngine(
                PiperTTSConfig(
                    model_path=cfg.tts_model_path,
                    speaker_id=0,
                    length_scale=cfg.tts_length_scale,
                    noise_scale=cfg.tts_noise_scale,
                    noise_w_scale=cfg.tts_noise_w_scale,
                    output_device=cfg.tts_output_device,
                    volume=cfg.tts_volume,
                )
            )

        # ---- Optional: STT (voice input)
        self.stt = None
        if cfg.enable_stt:
            from stt.streaming_stt import StreamingSTT, StreamingSTTConfig

            def on_partial(t: str):
                # Optional: show partials with proper line clearing
                if self.cfg.print_transcripts and t:
                    # Clear entire line and show partial
                    if self.cfg.debug_show_pipeline_status:
                        print(f"\r\033[2K📝 [TRANSCRIBING] {t}", end="", flush=True)
                    else:
                        print(f"\r\033[2K[PARTIAL] {t}", end="", flush=True)

            def on_final(t: str, audio_buffer=None):
                # Clear the partial line completely before showing final
                if self.cfg.print_transcripts:
                    print("\r\033[2K", end="", flush=True)
                self.handle_user_text(t, audio_buffer)
            
            def on_speech_start():
                if self.cfg.debug_show_pipeline_status:
                    print("\r\033[2K🗣️  [SPEECH DETECTED] Listening...", flush=True)
            
            def on_speech_end():
                if self.cfg.debug_show_pipeline_status:
                    print("\r\033[2K⏸️  [SILENCE] Processing utterance...", flush=True)

            stt_config = StreamingSTTConfig(
                # Audio settings
                sample_rate=cfg.audio_sample_rate,
                channels=cfg.audio_channels,
                frame_ms=cfg.audio_frame_ms,
                
                # VAD settings
                vad_aggressiveness=cfg.vad_aggressiveness,
                vad_energy_threshold=cfg.vad_energy_threshold,
                vad_smooth_window=cfg.vad_smooth_window,
                
                # Segmenter settings
                start_trigger_frames=cfg.start_trigger_frames,
                end_silence_ms=cfg.vad_silence_ms,
                min_utterance_ms=cfg.min_utterance_ms,
                max_utterance_ms=cfg.max_utterance_ms,
                
                # Rolling window
                rolling_window_sec=cfg.stt_rolling_window_sec,
                
                # STT inference
                infer_interval_ms=cfg.stt_infer_interval_ms,
                stt_model=cfg.stt_model,
                stt_device=cfg.stt_device,
                stt_compute_type=cfg.stt_compute_type,
                stt_beam_size=cfg.stt_beam_size,
                stt_language=cfg.stt_language,
                
                # Output behavior
                emit_partials=cfg.stt_emit_partials,
                partial_min_chars_delta=cfg.stt_partial_min_delta,
                
                # Debug
                debug_show_vad_state=cfg.debug_show_vad_state,
            )
            
            self.stt = StreamingSTT(
                stt_config,
                on_partial=on_partial,
                on_final=on_final,
                on_speech_start=on_speech_start,
                on_speech_end=on_speech_end,
            )

    def handle_user_text(self, user_text: str, audio_buffer=None):
        user_text = (user_text or "").strip()
        if not user_text:
            return

        if self.cfg.print_transcripts:
            if self.cfg.enable_stt:
                print(f"\n[USER/STT] {user_text}")
            else:
                print(f"[USER] {user_text}")
        
        # Log user input
        if self.logger:
            self.logger.log_user_input(user_text, audio_buffer)
        
        # Check for exit commands
        exit_keywords = ["goodbye", "bye", "exit", "quit", "stop", "shut down", "shutdown"]
        if any(keyword in user_text.lower() for keyword in exit_keywords):
            response = "Goodbye! Shutting down now."
            if self.cfg.print_llm_output:
                print(f"[ASSISTANT] {response}")
            if self.tts is not None:
                self.tts.speak(response)
            print("\nExit command detected. Shutting down...")
            self.shutdown()
            import sys
            sys.exit(0)

        # Update memory and generate response
        if self.cfg.debug_show_pipeline_status:
            print("🤖 [LLM THINKING] Generating response...", flush=True)
        
        self.memory.add_user(user_text)
        messages = self.prompt_mgr.build(user_text=user_text, history=self.memory.get())
        response = self.llm.generate(messages)
        response = self.prompt_mgr.postprocess(response)
        self.memory.add_assistant(response)

        if self.cfg.print_llm_output:
            print(f"[ASSISTANT] {response}")
        
        # Log assistant response
        if self.logger:
            self.logger.log_assistant_response(response)

        if self.tts is not None:
            if self.cfg.debug_show_pipeline_status:
                print("🔊 [SPEAKING] Playing audio...", flush=True)
            self.tts.speak(response)
            if self.cfg.debug_show_pipeline_status:
                print("🎤 [LISTENING] Ready for input...\n")
        
        # Add newline after response to separate from next input
        if self.cfg.enable_stt and not self.cfg.debug_show_pipeline_status:
            print()  # Blank line for readability

    def shutdown(self):
        """Clean shutdown of all components."""
        self._running = False
        if self.stt is not None:
            try:
                self.stt.stop()
            except Exception as e:
                print(f"Error stopping STT: {e}")
        if self.logger is not None:
            self.logger.close()

    def run(self):
        self._running = True

        # Ctrl+C handling
        def _sigint(_signum, _frame):
            print("\nShutting down...")
            self.shutdown()

        signal.signal(signal.SIGINT, _sigint)

        # If STT enabled, run voice loop
        if self.stt is not None:
            print("=" * 60)
            print("Mode: VOICE (STT -> LLM -> " + ("TTS" if self.tts else "PRINT") + ")")
            print("=" * 60)
            print("Speak into the microphone. Ctrl+C to stop.\n")
            if self.cfg.debug_show_pipeline_status:
                print("🎤 [LISTENING] Waiting for speech...\n")
            try:
                self.stt.run_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown()
            return

        # Otherwise fallback to typed input
        if not self.cfg.text_input_fallback:
            raise RuntimeError("ENABLE_STT=0 but TEXT_INPUT_FALLBACK=0. Nothing to run.")

        print("=" * 60)
        print("Mode: TEXT (INPUT -> LLM -> " + ("TTS" if self.tts else "PRINT") + ")")
        print("=" * 60)
        print("Type your message. Ctrl+C to stop.\n")

        try:
            while self._running:
                text = input("You> ").strip()
                if not text:
                    continue
                self.handle_user_text(text)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
            print("\nShutdown complete.")


def main():
    cfg = load_config()
    orch = Orchestrator(cfg)
    orch.run()


if __name__ == "__main__":
    main()
