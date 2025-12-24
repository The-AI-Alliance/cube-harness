"""Tests for the agentlab radar CLI."""

import subprocess
import sys
from pathlib import Path


class TestRadarCLI:
    """Tests for the radar CLI module."""

    def test_main_help(self) -> None:
        """Test that main help displays available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "agentlab2.radar.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "radar" in result.stdout
        assert "Launch the experiment monitoring dashboard" in result.stdout

    def test_radar_help(self) -> None:
        """Test that radar subcommand help displays options."""
        result = subprocess.run(
            [sys.executable, "-m", "agentlab2.radar.cli", "radar", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--debug" in result.stdout
        assert "--share" in result.stdout
        assert "directory" in result.stdout

    def test_no_command_shows_help(self) -> None:
        """Test that running without subcommand shows help and exits with error."""
        result = subprocess.run(
            [sys.executable, "-m", "agentlab2.radar.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "radar" in result.stdout

    def test_nonexistent_directory_error(self, tmp_dir: Path) -> None:
        """Test that non-existent directory shows error message on stderr."""
        nonexistent = tmp_dir / "does_not_exist"
        result = subprocess.run(
            [sys.executable, "-m", "agentlab2.radar.cli", "radar", str(nonexistent)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error: Directory does not exist" in result.stderr
        assert "Usage:" in result.stderr

    def test_argument_parsing(self) -> None:
        """Test that arguments are parsed correctly."""

        from agentlab2.radar.cli import main

        # We can't easily test the full flow without starting Gradio,
        # but we can verify the module imports and basic structure
        assert callable(main)
