# harkd

FastAPI daemon for meeting minutes, summaries, task extraction, transcription (WhisperX), and diarization (pyannote).

## Toolchain

- **Always use `uv run`** to run Python commands (`uv run pytest`, `uv run ruff`, etc.)
- All code (including tests) must pass `ruff check`, `ruff format`, and `pyrefly check` before committing
- Run tests: `uv run pytest tests/ -x --tb=short`
- Commits use **conventional commit** messages (`feat:`, `fix:`, `chore:`, etc.)

## Structure

Keep this section up-to-date when adding or removing top-level modules.

```
src/harkd/
  api/              # FastAPI routes + models
  audio/            # PortAudio recording (recorder.py, sources.py)
  chat/             # Agentic chat (Ask Hark) — tool calling, streaming, service
  services/         # Business logic (recording lifecycle, transcription)
  storage/          # Filesystem-based persistence (~/.local/share/hark/)
  llm/              # Optional LLM integration (meeting minutes, chat streaming)
  transcription/    # Pluggable transcription backends (local, koyeb, verda, scaleway)
  worker/           # Remote GPU worker FastAPI app (deployed as Docker container)
  config.py         # Pydantic settings (env: HARKD_* prefix)
worker/             # Dockerfile for GPU worker container
tests/              # pytest, mirrors src/ structure
```

## Audio recording pitfalls

- **Never do file I/O in PortAudio callbacks** — causes segfaults in async server context. Use a queue + writer thread.
- **Never map PulseAudio monitor names to ALSA devices** — use `PULSE_SOURCE` env var instead.
- **Never resample in audio callbacks** — too slow, causes buffer overflow.
