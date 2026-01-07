/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::vis::gui::panels {

    // Console message with color and type info
    struct ConsoleMessage {
        std::string text;
        uint32_t color;  // RGBA packed color
        bool is_input;   // True if this was user input
    };

    // Python console panel state (singleton)
    class PythonConsoleState {
    public:
        static PythonConsoleState& getInstance();

        // Add output from Python scripts
        void addOutput(const std::string& text, uint32_t color = 0xFFFFFFFF);
        void addError(const std::string& text);
        void addInput(const std::string& text);
        void clear();

        // Get messages for rendering
        const std::deque<ConsoleMessage>& messages() const { return messages_; }

        // Command history for up/down navigation
        void addToHistory(const std::string& cmd);
        std::string getHistoryEntry(int offset) const;
        void resetHistoryIndex() { history_index_ = -1; }
        void historyUp();
        void historyDown();
        int historyIndex() const { return history_index_; }

    private:
        PythonConsoleState() = default;

        std::deque<ConsoleMessage> messages_;
        std::vector<std::string> command_history_;
        int history_index_ = -1;
        mutable std::mutex mutex_;
        static constexpr size_t MAX_MESSAGES = 1000;
    };

    // Draw the Python console window
    void DrawPythonConsole(const UIContext& ctx, bool* open);

}  // namespace lfs::vis::gui::panels
