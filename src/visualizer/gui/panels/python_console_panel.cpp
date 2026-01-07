/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "gui/ui_widgets.hpp"

#include <imgui.h>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#include <mutex>

#include "python/runner.hpp"

namespace {
    std::once_flag g_console_init_once;

    void setup_console_output_capture() {
        std::call_once(g_console_init_once, [] {
            lfs::python::set_output_callback([](const std::string& text, bool is_error) {
                auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
                if (is_error) {
                    state.addError(text);
                } else {
                    state.addOutput(text);
                }
            });
        });
    }
}  // namespace
#endif

namespace lfs::vis::gui::panels {

    namespace {
        // Color constants (ImGui RGBA format - 0xAABBGGRR)
        constexpr uint32_t COLOR_INPUT = 0xFF88FF88;   // Green
        constexpr uint32_t COLOR_OUTPUT = 0xFFFFFFFF;  // White
        constexpr uint32_t COLOR_ERROR = 0xFF5555FF;   // Red
        constexpr uint32_t COLOR_INFO = 0xFFFFFF88;    // Yellow

        // Input buffer for command line
        char s_input_buffer[4096] = {};
        bool s_scroll_to_bottom = false;
        bool s_focus_input = true;
    }  // namespace

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
    }

    void PythonConsoleState::addOutput(const std::string& text, uint32_t color) {
        std::lock_guard lock(mutex_);
        messages_.push_back({text, color, false});
        while (messages_.size() > MAX_MESSAGES) {
            messages_.pop_front();
        }
    }

    void PythonConsoleState::addError(const std::string& text) {
        addOutput(text, COLOR_ERROR);
    }

    void PythonConsoleState::addInput(const std::string& text) {
        std::lock_guard lock(mutex_);
        messages_.push_back({"> " + text, COLOR_INPUT, true});
        while (messages_.size() > MAX_MESSAGES) {
            messages_.pop_front();
        }
    }

    void PythonConsoleState::clear() {
        std::lock_guard lock(mutex_);
        messages_.clear();
    }

    void PythonConsoleState::addToHistory(const std::string& cmd) {
        std::lock_guard lock(mutex_);
        if (!cmd.empty() && (command_history_.empty() || command_history_.back() != cmd)) {
            command_history_.push_back(cmd);
        }
        history_index_ = -1;
    }

    std::string PythonConsoleState::getHistoryEntry(int offset) const {
        std::lock_guard lock(mutex_);
        if (command_history_.empty()) return "";
        int idx = static_cast<int>(command_history_.size()) + offset;
        if (idx >= 0 && idx < static_cast<int>(command_history_.size())) {
            return command_history_[idx];
        }
        return "";
    }

    void PythonConsoleState::historyUp() {
        std::lock_guard lock(mutex_);
        if (command_history_.empty()) return;
        if (history_index_ < 0) {
            history_index_ = static_cast<int>(command_history_.size()) - 1;
        } else if (history_index_ > 0) {
            history_index_--;
        }
    }

    void PythonConsoleState::historyDown() {
        std::lock_guard lock(mutex_);
        if (history_index_ < 0) return;
        if (history_index_ < static_cast<int>(command_history_.size()) - 1) {
            history_index_++;
        } else {
            history_index_ = -1;
        }
    }

    void DrawPythonConsole(const UIContext& ctx, bool* open) {
        if (!open || !*open) return;

#ifndef LFS_BUILD_PYTHON_BINDINGS
        ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Python Console", open)) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "Python bindings not available. Rebuild with -DBUILD_PYTHON_BINDINGS=ON");
        }
        ImGui::End();
        return;
#else
        // Initialize Python and set up output capture
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();

        ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Python Console", open, ImGuiWindowFlags_MenuBar)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Console")) {
                if (ImGui::MenuItem("Clear")) {
                    state.clear();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Output area
        const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        if (ImGui::BeginChild("ConsoleOutput", ImVec2(0, -footer_height), true,
                              ImGuiWindowFlags_HorizontalScrollbar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 1));

            for (const auto& msg : state.messages()) {
                ImGui::PushStyleColor(ImGuiCol_Text, msg.color);
                ImGui::TextUnformatted(msg.text.c_str());
                ImGui::PopStyleColor();
            }

            if (s_scroll_to_bottom) {
                ImGui::SetScrollHereY(1.0f);
                s_scroll_to_bottom = false;
            }

            ImGui::PopStyleVar();
        }
        ImGui::EndChild();

        // Input line
        ImGui::Separator();

        // Handle history navigation
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && ImGui::IsItemActive()) {
            state.historyUp();
            if (state.historyIndex() >= 0) {
                const auto entry = state.getHistoryEntry(state.historyIndex() -
                                                         static_cast<int>(state.messages().size()));
                if (!entry.empty()) {
                    strncpy(s_input_buffer, entry.c_str(), sizeof(s_input_buffer) - 1);
                }
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && ImGui::IsItemActive()) {
            state.historyDown();
            if (state.historyIndex() >= 0) {
                const auto entry = state.getHistoryEntry(state.historyIndex() -
                                                         static_cast<int>(state.messages().size()));
                strncpy(s_input_buffer, entry.c_str(), sizeof(s_input_buffer) - 1);
            } else {
                s_input_buffer[0] = '\0';
            }
        }

        // Input text
        ImGui::PushItemWidth(-1);
        if (s_focus_input) {
            ImGui::SetKeyboardFocusHere();
            s_focus_input = false;
        }

        ImGuiInputTextFlags input_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                          ImGuiInputTextFlags_CallbackHistory;

        bool submit = ImGui::InputText("##PythonInput", s_input_buffer, sizeof(s_input_buffer), input_flags);
        ImGui::PopItemWidth();

        if (submit && s_input_buffer[0] != '\0') {
            std::string cmd(s_input_buffer);
            state.addInput(cmd);
            state.addToHistory(cmd);

            // Execute Python code
            PyGILState_STATE gil = PyGILState_Ensure();
            PyObject* result = PyRun_String(cmd.c_str(), Py_single_input,
                                            PyEval_GetGlobals(), PyEval_GetLocals());
            if (result) {
                if (result != Py_None) {
                    PyObject* str = PyObject_Str(result);
                    if (str) {
                        const char* output = PyUnicode_AsUTF8(str);
                        if (output && output[0] != '\0') {
                            state.addOutput(output);
                        }
                        Py_DECREF(str);
                    }
                }
                Py_DECREF(result);
            } else {
                PyErr_Print();
                state.addError("Error executing Python code");
            }
            PyGILState_Release(gil);

            s_input_buffer[0] = '\0';
            s_scroll_to_bottom = true;
            s_focus_input = true;
        }

        ImGui::End();
#endif  // LFS_BUILD_PYTHON_BINDINGS
    }

}  // namespace lfs::vis::gui::panels
