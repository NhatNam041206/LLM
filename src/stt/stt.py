from faster_whisper import WhisperModel
import time
import inspect
import json
from datetime import datetime
from typing import Iterable, Optional
import numpy as np


class STTRunner:
    """Encapsulate transcription modes and timing/logging.

    Two modes supported:
      - transcribe_file(audio_path): process an audio file end-to-end.
      - transcribe_stream(audio_chunks): process an iterable of numpy audio chunks (real-time).

    Both modes write text to `txt_file` and append a JSON summary line to `pred_stt_time`.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device_type: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 1,
        txt_file: str = "./preds/stt/pred.txt",
        pred_stt_time: str = "./logs/pred_stt_time.log.txt",
    ) -> None:
        self.model_size = model_size
        self.device_type = device_type
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.txt_file = txt_file
        self.pred_stt_time = pred_stt_time

        # Initialize model
        self.model = WhisperModel(self.model_size, device=self.device_type, compute_type=self.compute_type)

    def _call_transcribe(self, audio, **kwargs):
        """Call model.transcribe with backward-compatibility for older faster-whisper versions."""
        transcribe_sig = inspect.signature(self.model.transcribe)
        safe_kwargs = {}
        for k, v in kwargs.items():
            if k in transcribe_sig.parameters:
                safe_kwargs[k] = v
        return self.model.transcribe(audio, **safe_kwargs)

    def _process_segments(self, segments, out_file_handle) -> list:
        """Iterate over a segments generator and collect timing metadata.

        Returns list of segment timing dicts.
        """
        seg_iter = iter(segments)
        seg_idx = 0
        segment_timings = []

        while True:
            t0 = time.time()
            try:
                segment = next(seg_iter)
            except StopIteration:
                break
            t1 = time.time()

            production_time = t1 - t0
            audio_duration = getattr(segment, "end", 0) - getattr(segment, "start", 0)

            words_info = []
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    w_text = getattr(w, "word", None) or getattr(w, "text", None) or str(w)
                    w_start = getattr(w, "start", None)
                    w_end = getattr(w, "end", None)
                    if (w_start is not None) and (w_end is not None):
                        w_audio_dur = w_end - w_start
                    else:
                        w_audio_dur = audio_duration / max(1, len(segment.words))
                    words_info.append((w_text, w_start, w_end, w_audio_dur))
            else:
                tokens = [t for t in segment.text.strip().split() if t]
                per_word = audio_duration / max(1, len(tokens)) if tokens else 0
                for tkn in tokens:
                    words_info.append((tkn, None, None, per_word))

            # Write text
            out_file_handle.write(str(segment.text) + " ")

            segment_timings.append({
                "idx": seg_idx,
                "text": segment.text,
                "start": getattr(segment, "start", None),
                "end": getattr(segment, "end", None),
                "audio_duration": audio_duration,
                "production_time": production_time,
                "num_words": len(words_info),
                "words_info": words_info,
            })

            seg_idx += 1

        return segment_timings

    def _append_run_summary(self, run_summary: dict) -> None:
        try:
            with open(self.pred_stt_time, "a", encoding="utf-8") as tfile:
                tfile.write(json.dumps(run_summary) + "\n")
            print(f"Timing summary appended to '{self.pred_stt_time}'")
        except Exception as e:
            print(f"Warning: failed to write timing summary to '{self.pred_stt_time}': {e}")

    def transcribe_file(self, audio_path: str, beam_size: int = 1, **transcribe_options) -> dict:
        """Transcribe a file path and return a run summary.

        Example: runner.transcribe_file("file.mp3", beam_size=1, batch_size=2)
        """
        transcription_start = time.time()

        # call transcribe with safety
        segments, info = self._call_transcribe(audio_path, beam_size=beam_size, **transcribe_options)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        with open(self.txt_file, "w", encoding="utf-8") as f:
            segment_timings = self._process_segments(segments, f)

        transcription_end = time.time()
        total_transcription_time = transcription_end - transcription_start

        # aggregate
        if segment_timings:
            max_seg = max(segment_timings, key=lambda s: s["production_time"])
            min_seg = min(segment_timings, key=lambda s: s["production_time"])
            total_audio = sum(s["audio_duration"] for s in segment_timings)
            total_words = sum(s["num_words"] for s in segment_timings)
            avg_proc_per_word = sum(s["production_time"] for s in segment_timings) / max(1, total_words)

            run_summary = {
                "run_time": datetime.utcnow().isoformat() + "Z",
                "model": self.model_size,
                "device": self.device_type,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "beam_size": beam_size,
                "total_transcription_time": total_transcription_time,
                "total_audio": total_audio,
                "total_words": total_words,
                "avg_proc_per_word": avg_proc_per_word,
                "n_segments": len(segment_timings),
                "max_segment": {
                    "idx": max_seg["idx"],
                    "production_time": max_seg["production_time"],
                    "audio_duration": max_seg["audio_duration"],
                    "num_words": max_seg["num_words"],
                },
                "min_segment": {
                    "idx": min_seg["idx"],
                    "production_time": min_seg["production_time"],
                    "audio_duration": min_seg["audio_duration"],
                    "num_words": min_seg["num_words"],
                },
            }
        else:
            run_summary = {
                "run_time": datetime.utcnow().isoformat() + "Z",
                "model": self.model_size,
                "device": self.device_type,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "total_transcription_time": total_transcription_time,
                "total_audio": 0,
                "total_words": 0,
                "avg_proc_per_word": None,
                "n_segments": 0,
            }

        self._append_run_summary(run_summary)
        return run_summary

    def transcribe_stream(self, audio_chunks: Iterable[np.ndarray], beam_size: int = 1, chunk_sr: int = 16000, **transcribe_options) -> dict:
        """Process an iterable of raw audio chunks (numpy arrays at sample rate `chunk_sr`).

        This method treats each chunk as a short audio file and calls `transcribe` on it.
        It writes incremental results to `txt_file` and returns a summary after the stream ends.
        """
        transcription_start = time.time()
        total_audio = 0.0
        segment_timings_all = []

        # open file to append partial outputs
        with open(self.txt_file, "w", encoding="utf-8") as f:
            for chunk in audio_chunks:
                chunk_start = time.time()
                # chunk should be a 1-D numpy array with sample rate chunk_sr
                # call transcribe on the chunk
                segments, info = self._call_transcribe(chunk, beam_size=beam_size, **transcribe_options)
                # process segments; note production_time will measure iteration time inside _process_segments
                seg_timings = self._process_segments(segments, f)

                # approximate chunk duration
                chunk_dur = chunk.shape[0] / float(chunk_sr) if hasattr(chunk, "shape") else 0.0
                total_audio += chunk_dur
                segment_timings_all.extend(seg_timings)

        transcription_end = time.time()
        total_transcription_time = transcription_end - transcription_start

        # aggregate across stream
        if segment_timings_all:
            max_seg = max(segment_timings_all, key=lambda s: s["production_time"])
            min_seg = min(segment_timings_all, key=lambda s: s["production_time"])
            total_words = sum(s["num_words"] for s in segment_timings_all)
            avg_proc_per_word = sum(s["production_time"] for s in segment_timings_all) / max(1, total_words)

            run_summary = {
                "run_time": datetime.utcnow().isoformat() + "Z",
                "model": self.model_size,
                "device": self.device_type,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "beam_size": beam_size,
                "total_transcription_time": total_transcription_time,
                "total_audio": total_audio,
                "total_words": total_words,
                "avg_proc_per_word": avg_proc_per_word,
                "n_segments": len(segment_timings_all),
                "max_segment": {
                    "idx": max_seg["idx"],
                    "production_time": max_seg["production_time"],
                    "audio_duration": max_seg["audio_duration"],
                    "num_words": max_seg["num_words"],
                },
                "min_segment": {
                    "idx": min_seg["idx"],
                    "production_time": min_seg["production_time"],
                    "audio_duration": min_seg["audio_duration"],
                    "num_words": min_seg["num_words"],
                },
            }
        else:
            run_summary = {
                "run_time": datetime.utcnow().isoformat() + "Z",
                "model": self.model_size,
                "device": self.device_type,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "total_transcription_time": total_transcription_time,
                "total_audio": total_audio,
                "total_words": 0,
                "avg_proc_per_word": None,
                "n_segments": 0,
            }

        self._append_run_summary(run_summary)
        return run_summary


if __name__ == "__main__":
    # Simple CLI demo
    runner = STTRunner()
    input_path = "./data/A geometric notion of singularity.mp3"
    # Example: file mode
    if input_path:
        summary = runner.transcribe_file(input_path, beam_size=3, batch_size=3)
        print(summary)
    else:
        # Example: real-time mode - simulate with chunks from the same audio file
        try:
            import soundfile as sf

            data, sr = sf.read(input_path)
            # split into 5-second chunks
            chunk_len = 5 * sr
            chunks = [data[i : i + chunk_len] for i in range(0, len(data), chunk_len)]
            stream_summary = runner.transcribe_stream(chunks, beam_size=1, chunk_sr=sr, batch_size=1)
            print(stream_summary)
        except Exception:
            print("Skipping realtime demo: 'soundfile' not available or audio read failed.")
