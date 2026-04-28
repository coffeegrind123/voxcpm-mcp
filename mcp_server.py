# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio-client>=1.0,<2",
#     "mcp>=1.2.0,<2",
# ]
# ///

import base64
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from gradio_client import Client, handle_file
from mcp.server.fastmcp import FastMCP

DEFAULT_VOXCPM_URL = "http://host.docker.internal:8808"
VOICES_DIR = Path(__file__).resolve().parent / "voices"

LOG_LEVEL = os.getenv("VOXCPM_MCP_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("voxcpm-mcp")

voxcpm_url = os.getenv("VOXCPM_URL", DEFAULT_VOXCPM_URL).rstrip("/")

mcp = FastMCP("voxcpm")

_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        logger.info(f"Connecting to VoxCPM at {voxcpm_url}")
        _client = Client(voxcpm_url)
    return _client


def _voice_path(name: str) -> Path:
    """Resolve a voice preset name to a file path."""
    p = VOICES_DIR / name
    if p.suffix not in (".mp3", ".wav", ".flac", ".ogg"):
        for ext in (".mp3", ".wav"):
            candidate = VOICES_DIR / f"{name}{ext}"
            if candidate.exists():
                return candidate
    if not p.exists():
        raise FileNotFoundError(f"Voice preset '{name}' not found in {VOICES_DIR}")
    return p


def _resolve_audio_path(value: str) -> str:
    """Resolve audio input: voice preset name, local file path, or base64.

    Returns a path that can be passed to handle_file().
    """
    v = value.strip()
    if not v:
        return ""

    # 1. Voice preset name (no extension, no path separators, no base64)
    if "/" not in v and "\\" not in v and "," not in v:
        try:
            return str(_voice_path(v))
        except FileNotFoundError:
            pass

    # 2. Local file path
    p = Path(v)
    if p.exists():
        return str(p)

    # 3. Base64 data URI or raw base64
    try:
        clean = v.split(",", 1)[1] if "," in v else v
        raw = base64.b64decode(clean, validate=True)
        suffix = ".wav"
        if raw[:4] == b"RIFF":
            suffix = ".wav"
        elif raw[:3] == b"ID3" or (raw[0] == 0xFF and raw[1] == 0xFB):
            suffix = ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(raw)
        tmp.close()
        logger.info(f"Decoded base64 audio to {tmp.name} ({len(raw)} bytes)")
        return tmp.name
    except Exception:
        pass

    # 4. Give up
    raise FileNotFoundError(
        f"Could not resolve audio input '{value[:80]}'. "
        "Must be a voice preset name, a local file path, or base64-encoded audio."
    )


# =============================================================================
# MCP Tools — 1:1 bridge to Gradio API
# =============================================================================

@mcp.tool()
def generate(
    text: str = "VoxCPM2 is a creative multilingual TTS model from ModelBest, designed to generate highly realistic speech.",
    control_instruction: str = "",
    ref_wav: str = "",
    use_prompt_text: bool = False,
    prompt_text_value: str = "",
    cfg_value: float = 2.0,
    do_normalize: bool = False,
    denoise: bool = False,
    dit_steps: float = 10.0,
) -> str:
    """Generate speech from text using VoxCPM2.

    Direct bridge to the Gradio /generate endpoint. Accepts the exact same
    parameters as the VoxCPM web UI.

    Three modes depending on inputs:
    - Voice Design: text + control_instruction (no ref_wav)
    - Controllable Cloning: text + ref_wav + optional control_instruction
    - Ultimate Cloning: text + ref_wav + prompt_text_value (set use_prompt_text=True)

    Args:
        text: The text to synthesize into speech.
        control_instruction: Voice description — gender, age, tone, emotion, pace.
            Supports Chinese & English. Ignored when use_prompt_text=True.
            Examples: "A young girl with a soft sweet voice, speaking slowly"
                     "暴躁的中年男声，语速快，充满无奈和愤怒"
        ref_wav: Reference audio for cloning. Can be:
            - A voice preset name from list_voice_presets() (e.g. "airy", "mellow")
            - An absolute path to a local audio file (WAV/MP3/FLAC)
            - Base64-encoded audio (raw base64 or data:audio/... URI)
        use_prompt_text: Set True for Ultimate Cloning mode.
            Disables control_instruction.
        prompt_text_value: Transcript of the reference audio.
            Required when use_prompt_text=True.
        cfg_value: Guidance scale 1.0-3.0. Higher = closer to reference/prompt.
            Lower = more creative variation.
        do_normalize: Normalize numbers, dates, abbreviations in text via wetext.
        denoise: Apply ZipEnhancer denoising to reference audio before cloning.
        dit_steps: LocDiT flow-matching steps 1-50. More = potentially better
            quality but slower.

    Returns:
        JSON string with keys:
          - success (bool)
          - audio (str): base64 data URI of the generated WAV
          - size_bytes (int): size of the WAV in bytes
          - error (str): error message if success is False
    """
    try:
        client = _get_client()

        # Resolve ref_wav to a file path that handle_file() can upload
        resolved = _resolve_audio_path(ref_wav) if ref_wav else ""

        result = client.predict(
            text=text,
            control_instruction=control_instruction,
            ref_wav=handle_file(resolved) if resolved else None,
            use_prompt_text=use_prompt_text,
            prompt_text_value=prompt_text_value,
            cfg_value=float(cfg_value),
            do_normalize=bool(do_normalize),
            denoise=bool(denoise),
            dit_steps=float(dit_steps),
            api_name="/generate",
        )

        # Result is a Gradio temp path. Extract filename and point to
        # the persistent copy at /app/outputs/ (kept for 10 generations).
        raw = str(result[-1]) if isinstance(result, (tuple, list)) else str(result)
        filename = Path(raw).name  # gen_1777393953503.wav
        audio_url = f"{voxcpm_url}/gradio_api/file=/app/outputs/{filename}"

        logger.info(f"Generated audio: {audio_url}")

        return json.dumps({
            "success": True,
            "audio_url": audio_url,
        })

    except Exception as e:
        logger.error(f"generate failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
def run_asr(
    checked: bool = True,
    audio_path: str = "",
) -> str:
    """Transcribe audio to text using SenseVoiceSmall ASR.

    Direct bridge to the Gradio /_run_asr_if_needed endpoint.

    Args:
        checked: Set True to run ASR (required).
        audio_path: Audio to transcribe. Can be:
            - An absolute path to a local audio file (WAV/MP3/FLAC)
            - Base64-encoded audio (raw base64 or data:audio/... URI)

    Returns:
        JSON string with keys:
          - success (bool)
          - text (str): transcribed text
          - error (str): error message if success is False
    """
    try:
        resolved = _resolve_audio_path(audio_path) if audio_path else ""
        if not resolved:
            return json.dumps({"success": False, "error": "audio_path is required"})

        client = _get_client()
        result = client.predict(
            checked=checked,
            audio_path=handle_file(resolved),
            api_name="/_run_asr_if_needed",
        )

        # Gradio returns {'value': '...', '__type__': 'update'} for Textbox
        if isinstance(result, dict):
            text = str(result.get("value", ""))
        else:
            text = str(result) if result else ""

        return json.dumps({"success": True, "text": text})

    except Exception as e:
        logger.error(f"run_asr failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
def toggle_ultimate_cloning(checked: bool = False) -> str:
    """Toggle Ultimate Cloning mode in the UI state.

    Direct bridge to the Gradio /_on_toggle_instant endpoint.
    When checked=True, control_instruction is hidden and prompt_text is shown.

    Returns:
        JSON string with keys:
          - success (bool)
          - prompt_text_visible (bool)
          - control_visible (bool)
    """
    try:
        client = _get_client()
        _ = client.predict(checked=checked, api_name="/_on_toggle_instant")
        return json.dumps({
            "success": True,
            "prompt_text_visible": checked,
            "control_visible": not checked,
        })
    except Exception as e:
        logger.error(f"toggle_ultimate_cloning failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


# =============================================================================
# Convenience tools
# =============================================================================

@mcp.tool()
def list_voice_presets() -> str:
    """List available voice preset files for cloning.

    These are pre-recorded reference voices in the voices/ directory.
    Use the returned names as the ref_wav parameter in generate().

    Returns:
        JSON string with keys:
          - success (bool)
          - presets (list[str]): available voice preset names
    """
    try:
        presets = sorted(
            p.stem for p in VOICES_DIR.iterdir()
            if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg")
        )
        return json.dumps({"success": True, "presets": presets})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
def ping() -> str:
    """Check connectivity to the VoxCPM Gradio server."""
    try:
        client = _get_client()
        client.predict(checked=False, api_name="/_on_toggle_instant")
        return json.dumps({"success": True, "url": voxcpm_url, "alive": True})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "url": voxcpm_url})


# =============================================================================
# Entry point
# =============================================================================

def _wait_for_backend(timeout: int = 600) -> bool:
    """Poll the Gradio backend until it responds or times out.

    Returns True if ready, False if timeout.
    """
    from urllib.request import urlopen, Request
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            req = Request(f"{voxcpm_url}/gradio_api/info")
            with urlopen(req, timeout=5) as r:
                if r.status == 200:
                    logger.info(f"Backend ready after {attempt} attempt(s)")
                    return True
        except Exception:
            pass
        remaining = int(deadline - time.time())
        logger.info(
            f"Waiting for VoxCPM backend at {voxcpm_url} "
            f"(attempt {attempt}, {remaining}s remaining)..."
        )
        time.sleep(5)
    logger.error(f"Backend did not become ready within {timeout}s")
    return False


if __name__ == "__main__":
    logger.info(f"VoxCPM MCP Server starting, backend: {voxcpm_url}")
    logger.info(f"Voice presets dir: {VOICES_DIR} ({'exists' if VOICES_DIR.is_dir() else 'MISSING'})")

    if not _wait_for_backend():
        logger.error("Backend not ready, exiting. Claude Code will retry.")
        sys.exit(1)

    mcp.run()
