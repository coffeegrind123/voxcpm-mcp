# VoxCPM MCP Server

An MCP server that bridges [VoxCPM2](https://github.com/OpenBMB/VoxCPM) — a state-of-the-art multilingual TTS model — to any MCP client (Claude Code, etc.). The server wraps VoxCPM's Gradio API as MCP tools, letting LLMs generate speech, clone voices, and transcribe audio through a clean programmatic interface.

```
LLM (Claude Code)
  └─ stdio ─► mcp_server.py              (Python, FastMCP)
                └─ gradio_client ─► VoxCPM Gradio app (Docker, GPU, port 8808)
                                     ├── VoxCPM2 (TTS, 4.58GB, CUDA bf16)
                                     ├── ZipEnhancer (denoising)
                                     └── SenseVoiceSmall (ASR)
```

## Feature Surface

- 5 MCP tools covering all 3 Gradio API endpoints
- **Voice Design** — create new voices from text descriptions (gender, age, tone, emotion, pace)
- **Controllable Cloning** — clone a voice with optional style guidance
- **Ultimate Cloning** — reproduce every vocal nuance via audio continuation
- **ASR transcription** — SenseVoiceSmall for multilingual speech-to-text
- **12 voice presets** — pre-recorded reference voices for instant cloning
- Auto-resolving audio inputs: voice preset names, local file paths, or base64 data URIs
- Persistent audio output with rotation (last 10 generations kept)

See `mcp_server.py` for the complete tool inventory with docstrings.

## Requirements

**Host (where VoxCPM runs):**
- Docker with NVIDIA GPU support (nvidia-container-toolkit)
- VoxCPM Gradio app running on port 8808 (see [VoxCPM](https://github.com/OpenBMB/VoxCPM))
- ~10GB GPU VRAM for VoxCPM2

**MCP client host (where this server runs):**
- Python 3.10+
- `gradio-client` and `mcp` packages

## Installation

```bash
git clone https://github.com/coffeegrind123/voxcpm-mcp
cd voxcpm-mcp
pip install -r requirements.txt
```

Register the MCP server with Claude Code:

```bash
claude mcp add voxcpm -s user -- /bin/bash /path/to/voxcpm-mcp/run-mcp.sh
```

Or edit `~/.claude/.claude.json` directly:

```json
{
  "mcpServers": {
    "voxcpm": {
      "command": "/bin/bash",
      "args": ["/path/to/voxcpm-mcp/run-mcp.sh"]
    }
  }
}
```

The bridge connects to `http://host.docker.internal:8808` by default. Override with `VOXCPM_URL` if the Gradio app runs on a different host or port:

```bash
export VOXCPM_URL=http://192.168.1.10:8808
```

## VoxCPM Container Setup

The VoxCPM Gradio app must be running in Docker. From the VoxCPM repo:

```bash
git clone https://github.com/coffeegrind123/VoxCPM
cd VoxCPM
docker compose up -d
```

The container handles model downloads, warmup (torch.compile), and serves the Gradio API on port 8808. The MCP server connects to it via HTTP — no shared filesystem needed.

## Project Layout

```
.
├── mcp_server.py           # FastMCP server, 5 MCP tools
├── run-mcp.sh              # Startup wrapper (sets VOXCPM_URL)
└── requirements.txt    # gradio-client, mcp
```

## MCP Tools

### `generate`
1:1 bridge to Gradio `/generate`. 9 parameters matching the web UI exactly.

| Param | Type | Default | Description |
|---|---|---|---|
| `text` | str | — | Text to synthesize |
| `control_instruction` | str | `""` | Voice description (gender, age, tone, emotion, pace). Chinese & English. |
| `ref_wav` | str | `""` | Reference audio: preset name, file path, or base64 data URI |
| `use_prompt_text` | bool | `False` | Enable Ultimate Cloning mode |
| `prompt_text_value` | str | `""` | Transcript of reference audio (for Ultimate Cloning) |
| `cfg_value` | float | `2.0` | Guidance scale 1.0–3.0 |
| `do_normalize` | bool | `False` | Normalize numbers/dates via wetext |
| `denoise` | bool | `False` | ZipEnhancer denoising on reference audio |
| `dit_steps` | float | `10.0` | LocDiT flow-matching steps 1–50 |

Returns: `{"success": true, "audio_url": "http://.../gradio_api/file=/app/outputs/gen_xxx.wav"}`

### `run_asr`
1:1 bridge to Gradio `/_run_asr_if_needed`. Transcribe audio to text.

| Param | Type | Default | Description |
|---|---|---|---|
| `checked` | bool | `True` | Must be True to run ASR |
| `audio_path` | str | `""` | Audio to transcribe (path or base64) |

### `toggle_ultimate_cloning`
1:1 bridge to Gradio `/_on_toggle_instant`. UI state toggle.

### `list_voice_presets`
Returns available voice preset names for use as `ref_wav`.

### `ping`
Health check — verifies connectivity to the VoxCPM Gradio server.

## Voice Presets

12 pre-recorded reference voices in the VoxCPM repo's `voices/` directory:

| Preset | Character |
|---|---|
| `airy` | Light, breathy, ethereal |
| `buttery` | Smooth, rich, warm |
| `disconnected` | Detached, flat, robotic |
| `enter_voice_mode` | System prompt — entering voice mode |
| `exit_voice_mode` | System prompt — exiting voice mode |
| `final` | Authoritative, conclusive, bold |
| `glassy` | Clear, crisp, brittle |
| `intro` | Opening/narrative tone |
| `pre_recommendations` | Recommendation lead-in |
| `pre_voice` | Voice mode preamble |
| `recommendations` | Suggestive, advisory tone |
| `rounded` | Full, warm, balanced |

## Three TTS Modes

### Voice Design
No reference audio. Describe the voice and VoxCPM2 creates it.

```
generate(text="Hello world", control_instruction="A warm maternal voice, gentle and reassuring")
```

### Controllable Cloning
Upload reference audio, optionally add style guidance.

```
generate(text="Hello world", ref_wav="airy")
generate(text="Hello world", ref_wav="/path/to/recording.wav", control_instruction="Faster and more energetic")
```

### Ultimate Cloning
Provide the transcript of the reference audio for full vocal nuance preservation.

```
run_asr(audio_path="/path/to/reference.wav")           # get transcript first
generate(text="Hello world", ref_wav="/path/to/reference.wav",
         use_prompt_text=True, prompt_text_value="the transcript here")
```

## Audio Output

Generated audio is persisted to `/app/outputs/gen_*.wav` inside the VoxCPM container (last 10 kept). The `audio_url` in the response points to a Gradio-served WAV file that persists across requests.

## Agent Skill

A companion Claude Code skill (`voice-gen`) auto-invokes on speech/TTS keywords. Install it:

```bash
mkdir -p ~/.claude/skills/voice-gen
# See the voxcpm-mcp repo or VoxCPM repo for the SKILL.md
```

The skill handles preset selection, control instruction crafting, and parameter tuning automatically.

## Architecture Notes

### Bridge pattern

The MCP server is a pure bridge — it doesn't load models or do inference. All TTS/ASR work happens inside the VoxCPM Docker container, which the MCP server calls via `gradio_client`. This keeps the MCP server lightweight (~100KB) and allows it to run anywhere with network access to the Gradio app.

### Audio input resolution

The `ref_wav` and `audio_path` parameters accept three formats, auto-detected:
1. **Voice preset name** (no slashes, no base64 markers) → resolved from the `voices/` directory
2. **Local file path** → checked for existence on the MCP host, uploaded to Gradio via `handle_file()`
3. **Base64 data URI** → decoded to a temp file on the MCP host, uploaded to Gradio

### Persistent output

The Gradio app saves every generation to `/app/outputs/gen_{timestamp}.wav` and rotates to keep the 10 most recent files. The MCP server extracts the filename from Gradio's response and constructs a persistent URL pointing to `/app/outputs/`. The `allowed_paths` Gradio config ensures these files are served.

## Reference Implementations

- **MCP bridge pattern**: [cheat-engine-mcp](https://github.com/coffeegrind123/cheat-engine-mcp) — same FastMCP + HTTP bridge architecture
- **Gradio API bridge**: [GhidraMCP](https://github.com/LaurieWired/GhidraMCP) — dual-process HTTP bridge pattern
- **VoxCPM upstream**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — the TTS model this server wraps

## Troubleshooting

**`Server disconnected without sending a response`**
The VoxCPM container is still warming up models. Wait ~2 minutes after `docker compose up -d` for torch.compile warmup to complete.

**`audio_url` returns 403 Forbidden**
The Gradio container needs `allowed_paths=["/app/outputs"]` in its launch config. This is included in the VoxCPM fork at `coffeegrind123/VoxCPM`.

**`File not found` for voice presets**
The `voices/` directory must exist alongside the MCP server or in the VoxCPM repo. Presets are resolved relative to `mcp_server.py`'s location.

**`Cannot reach VoxCPM at http://host.docker.internal:8808`**
- Verify the VoxCPM container is running: `docker ps | grep voxcpm`
- From inside another container, use `host.docker.internal` (not `localhost`)
- Set `VOXCPM_URL` if the Gradio app is on a different host
