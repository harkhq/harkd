# harkd

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> meeting minutes, summaries, transcription & diarization, and task extraction daemon with REST API

harkd is a background service that records audio, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and exposes everything over a local REST API. Optional LLM integrations enable meeting minutes, summaries, and task extraction. It runs 100% offline by default - no cloud, no API keys, no data leaves your machine.

## Features

- 🔊 **Multi-source audio capture** - microphone, system audio, or both simultaneously
- ✨ **High-accuracy transcription** via faster-whisper (tiny through large-v3)
- 🗣️ **Speaker diarization** - identify and label who said what via [WhisperX](https://github.com/m-bain/whisperX) / [pyannote](https://github.com/pyannote/pyannote-audio)
- 🔇 **Audio preprocessing** - noise reduction and normalization
- 🤖 **LLM integration** - meeting minutes, summaries, and task extraction (OpenAI, Anthropic, Google, Ollama)
- 👤 **Voice profiles** - persistent speaker identification across recordings
- 🔒 **100% offline** - all processing happens locally, nothing leaves your machine
- 🌍 **Multilingual** - automatic language detection or explicit language selection

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/harkhq/harkd/main/install.sh | sh
```

This will:
- Install system dependencies (portaudio, ffmpeg) via your package manager
- Install [uv](https://docs.astral.sh/uv/) if not already present
- Install harkd from GitHub
- Register harkd as a background service (systemd on Linux, launchd on macOS)

### Service Management

**Linux:**

```bash
systemctl --user status harkd     # check status
journalctl --user -u harkd -f     # view logs
systemctl --user restart harkd    # restart
systemctl --user stop harkd       # stop
```

**macOS:**

```bash
launchctl print gui/$(id -u)/com.harkhq.harkd        # check status
tail -f ~/Library/Logs/harkd/stderr.log               # view logs
launchctl kickstart -k gui/$(id -u)/com.harkhq.harkd  # restart
launchctl kill SIGTERM gui/$(id -u)/com.harkhq.harkd   # stop
```

### Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/harkhq/harkd/main/install.sh | sh -s -- --uninstall
```

## API

harkd listens on `http://localhost:8765` by default. All endpoints are under `/api/v1`.

```bash
# Health check
curl http://localhost:8765/api/v1/health
```

Full API documentation is available via the built-in Swagger UI at [localhost:8765/docs](http://localhost:8765/docs) and ReDoc at [localhost:8765/redoc](http://localhost:8765/redoc).

## Configuration

harkd reads `~/.config/hark/daemon.yaml` on startup. Settings can also be set via environment variables with the `HARKD_` prefix and `__` as a nested delimiter.

```yaml
# ~/.config/hark/daemon.yaml

server:
  host: "127.0.0.1"
  port: 8765
  reload: false  # auto-reload on code changes (dev only)

cors:
  enabled: true
  origins:
    - "http://localhost:5173"
    - "http://localhost:3000"
    - "moz-extension://*"

storage:
  base_path: "~/.local/share/hark"

logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: null   # optional log file path

recording:
  model: large-v3       # tiny, base, small, medium, large, large-v2, large-v3
  language: auto         # language code or "auto"
  diarization: true      # enable speaker diarization
  noise_reduction: true  # enable noise reduction
  normalization: true    # enable audio normalization
  word_timestamps: false # include word-level timestamps

llm:
  enabled: false
  provider: openai  # openai, anthropic, google, ollama
  model: gpt-4o-mini
  api_key: null     # not needed for ollama
  base_url: null    # custom endpoint (e.g. ollama)
  temperature: 0.3
  max_tokens: 4096
  enable_cache: false
  enable_logging: true
  enable_token_stats: true
  custom_prompts_dir: null  # directory with custom prompt .txt files

# HuggingFace token for pyannote diarization models
hf_token: null
```

**Environment variable examples:**

```bash
HARKD_SERVER__PORT=9000
HARKD_RECORDING__MODEL=base
HARKD_RECORDING__DIARIZATION=false
HARKD_LLM__ENABLED=true
HARKD_LLM__PROVIDER=ollama
HARKD_LLM__BASE_URL=http://localhost:11434
HARKD_HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Transcription Backends

By default harkd transcribes locally. To offload to a GPU server, deploy the worker container and point harkd at it.

```bash
# Build and push the worker image
docker build -f worker/Dockerfile -t ghcr.io/<you>/harkd/worker .
docker push ghcr.io/<you>/harkd/worker
```

| Backend | `backend:` | Provider | How it works |
|---|---|---|---|
| **Local** | `local` | — | Runs WhisperX as a subprocess on your machine. No config needed. |
| **Koyeb** | `koyeb` | [Koyeb](https://koyeb.com) | Sync HTTP POST. Requires `koyeb.token` and `endpoint_url`. |
| **Scaleway** | `scaleway` | [Scaleway](https://scaleway.com) | Sync HTTP POST. Also works as a generic backend for any self-hosted worker. |
| **Verda** | `verda` | [DataCrunch](https://datacrunch.io) | Async job submission + polling. Optionally uses S3 for file transfer. |

All remote backends use the same worker image. Set `fallback_to_local: true` to fall back automatically on remote failure.

```yaml
# Example: Koyeb backend
transcription:
  backend: koyeb
  endpoint_url: https://your-worker.koyeb.app
  worker_api_key: your-shared-secret  # must match HARKD_WORKER_API_KEY on the worker
  fallback_to_local: true
  koyeb:
    token: your-koyeb-api-token
```

### Speaker Diarization

Diarization requires a HuggingFace token for [pyannote](https://github.com/pyannote/pyannote-audio) models:

1. Create an account at https://huggingface.co
2. Accept the model licenses:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create a token at https://huggingface.co/settings/tokens
4. Set `hf_token` in `daemon.yaml` or via `HARKD_HF_TOKEN`

### System Audio Capture

Recording system audio (speaker input) works differently per platform:

**Linux (PulseAudio/PipeWire):** Uses monitor sources automatically. Verify with:

```bash
pactl list sources | grep -i monitor
```

**macOS:** Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual audio driver:

```bash
brew install blackhole-2ch
```

Then create a Multi-Output Device in Audio MIDI Setup that includes both your speakers and BlackHole.

## Development

```bash
git clone https://github.com/harkhq/harkd.git
cd harkd
uv sync --extra test
uv run pre-commit install
uv run pytest
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the [**AGPLv3 License**](LICENSE).

## Acknowledgments

This project would not exist without the hard work of others, first and foremost the maintainers and contributors of the below mentioned projects:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
