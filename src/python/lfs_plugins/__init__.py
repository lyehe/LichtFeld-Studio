# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LichtFeld Plugin System."""

from .errors import (
    PluginDependencyError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginVersionError,
    RegistryError,
    RegistryOfflineError,
    VersionNotFoundError,
)
from .manager import PluginManager
from .panels import PluginManagerPanel, register_builtin_panels
from .plugin import PluginInfo, PluginInstance, PluginState
from .registry import RegistryClient, RegistryPluginInfo, RegistryVersionInfo

__all__ = [
    "PluginManager",
    "PluginInfo",
    "PluginState",
    "PluginInstance",
    "PluginError",
    "PluginLoadError",
    "PluginDependencyError",
    "PluginVersionError",
    "RegistryError",
    "RegistryOfflineError",
    "PluginNotFoundError",
    "VersionNotFoundError",
    "RegistryClient",
    "RegistryPluginInfo",
    "RegistryVersionInfo",
    "PluginManagerPanel",
    "register_builtin_panels",
]
