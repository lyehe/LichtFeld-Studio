/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace lfs::vis {
    // Forward declarations
    class VisualizerImpl;
    class EditorContext;

    namespace gui {
        class FileBrowser;

        // Shared context passed to all UI functions
        struct UIContext {
            VisualizerImpl* viewer;
            FileBrowser* file_browser;
            std::unordered_map<std::string, bool>* window_states;
            EditorContext* editor;  // Centralized editor state
        };
    } // namespace gui
} // namespace lfs::vis