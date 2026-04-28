#!/bin/bash
# Startup wrapper for the VoxCPM MCP server.
#
# Override VOXCPM_URL if the Gradio app isn't reachable at
# the default host.docker.internal:8808.

: "${VOXCPM_URL:=http://host.docker.internal:8808}"
export VOXCPM_URL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[VoxCPM MCP] Backend: ${VOXCPM_URL}" >&2
exec python3 "${SCRIPT_DIR}/mcp_server.py"
