#!/bin/bash
# Claude-style HUD integration script for shells
# Usage: Add $(claude-hud) to your shell prompt or tmux status line

# Check if mullet-claude-hud is installed
if command -v mullet-claude-hud &> /dev/null; then
    mullet-claude-hud
else
    echo "● ?  CLI not found"
fi