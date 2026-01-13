# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin template generator for scaffolding new plugins."""

import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

PLUGIN_TOML = '''[plugin]
name = "{name}"
version = "0.1.0"
description = "A new LichtFeld plugin"
author = ""

[dependencies]
packages = []

[lifecycle]
auto_start = true
hot_reload = true
'''

INIT_PY = '''"""
{name} - A LichtFeld Studio plugin.
"""

import lichtfeld as lf

_panel = None


def on_load():
    """Called when plugin is loaded."""
    global _panel
    from .panels.main_panel import MainPanel

    _panel = MainPanel
    lf.plugins.register_panel(MainPanel)
    lf.log.info("{name} plugin loaded")


def on_unload():
    """Called when plugin is unloaded."""
    global _panel
    if _panel:
        lf.plugins.unregister_panel(_panel)
        _panel = None
    lf.log.info("{name} plugin unloaded")
'''

MAIN_PANEL_PY = '''"""Main panel for {name} plugin."""

import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    """Example plugin panel."""

    label = "{title}"
    category = "Plugins"

    def __init__(self):
        super().__init__()

    def draw(self):
        lf.ui.text("Hello from {name}!")

        if lf.ui.button("Click Me"):
            lf.log.info("{name}: Button clicked!")
'''


def create_plugin(name: str, target_dir: Optional[Path] = None) -> Path:
    """Create a new plugin from template.

    Args:
        name: Plugin name (used for directory and module)
        target_dir: Optional target directory (defaults to ~/.lichtfeld/plugins)

    Returns:
        Path to created plugin directory

    Raises:
        FileExistsError: If plugin directory already exists
    """
    if target_dir is None:
        target_dir = Path.home() / ".lichtfeld" / "plugins"

    plugin_dir = target_dir / name
    if plugin_dir.exists():
        raise FileExistsError(f"Plugin directory already exists: {plugin_dir}")

    title = name.replace("_", " ").title()

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "panels").mkdir(exist_ok=True)

    (plugin_dir / "plugin.toml").write_text(PLUGIN_TOML.format(name=name))
    (plugin_dir / "__init__.py").write_text(INIT_PY.format(name=name))
    (plugin_dir / "panels" / "__init__.py").write_text("")
    (plugin_dir / "panels" / "main_panel.py").write_text(
        MAIN_PANEL_PY.format(name=name, title=title)
    )

    _log.info("Created plugin template at %s", plugin_dir)
    return plugin_dir
