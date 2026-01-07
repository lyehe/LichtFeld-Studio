/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_scripts_panel.hpp"
#include "gui/ui_widgets.hpp"

#include <imgui.h>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include "python/runner.hpp"
#endif

namespace lfs::vis::gui::panels {

    PythonScriptManagerState& PythonScriptManagerState::getInstance() {
        static PythonScriptManagerState instance;
        return instance;
    }

    void PythonScriptManagerState::addScript(const std::filesystem::path& path) {
        std::lock_guard lock(mutex_);
        for (const auto& s : scripts_) {
            if (s.path == path)
                return;
        }
        scripts_.push_back({path, true, false, ""});
    }

    void PythonScriptManagerState::setScripts(const std::vector<std::filesystem::path>& paths) {
        std::lock_guard lock(mutex_);
        scripts_.clear();
        for (const auto& p : paths) {
            scripts_.push_back({p, true, false, ""});
        }
    }

    void PythonScriptManagerState::setScriptEnabled(size_t index, bool enabled) {
        std::lock_guard lock(mutex_);
        if (index < scripts_.size()) {
            scripts_[index].enabled = enabled;
        }
    }

    void PythonScriptManagerState::setScriptError(size_t index, const std::string& error) {
        std::lock_guard lock(mutex_);
        if (index < scripts_.size()) {
            scripts_[index].has_error = !error.empty();
            scripts_[index].error_message = error;
        }
    }

    void PythonScriptManagerState::clearErrors() {
        std::lock_guard lock(mutex_);
        for (auto& s : scripts_) {
            s.has_error = false;
            s.error_message.clear();
        }
    }

    void PythonScriptManagerState::clear() {
        std::lock_guard lock(mutex_);
        scripts_.clear();
        needs_reload_ = false;
    }

    std::vector<std::filesystem::path> PythonScriptManagerState::enabledScripts() const {
        std::lock_guard lock(mutex_);
        std::vector<std::filesystem::path> result;
        for (const auto& s : scripts_) {
            if (s.enabled) {
                result.push_back(s.path);
            }
        }
        return result;
    }

    void DrawPythonScriptsPanel(const UIContext& ctx, bool* open) {
        (void)ctx;
        if (!open || !*open)
            return;

#ifndef LFS_BUILD_PYTHON_BINDINGS
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Python Scripts", open)) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "Python bindings not available. Rebuild with -DBUILD_PYTHON_BINDINGS=ON");
        }
        ImGui::End();
        return;
#else
        auto& state = PythonScriptManagerState::getInstance();

        ImGui::SetNextWindowSize(ImVec2(450, 300), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Python Scripts", open, ImGuiWindowFlags_MenuBar)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Actions")) {
                if (ImGui::MenuItem("Reload All", nullptr, false, !state.scripts().empty())) {
                    state.clearErrors();
                    auto scripts = state.enabledScripts();
                    if (!scripts.empty()) {
                        auto result = lfs::python::run_scripts(scripts);
                        if (!result) {
                            // Mark all as error if general failure
                            for (size_t i = 0; i < state.scripts().size(); ++i) {
                                if (state.scripts()[i].enabled) {
                                    state.setScriptError(i, result.error());
                                }
                            }
                        }
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Clear All")) {
                    state.clear();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Script list
        if (state.scripts().empty()) {
            ImGui::TextDisabled("No Python scripts loaded.");
            ImGui::TextDisabled("Use --python-script <path> to load scripts.");
        } else {
            ImGui::Text("Loaded Scripts:");
            ImGui::Separator();

            for (size_t i = 0; i < state.scripts().size(); ++i) {
                const auto& script = state.scripts()[i];
                ImGui::PushID(static_cast<int>(i));

                // Checkbox for enable/disable
                bool enabled = script.enabled;
                if (ImGui::Checkbox("##enabled", &enabled)) {
                    state.setScriptEnabled(i, enabled);
                }
                ImGui::SameLine();

                // Script name with color based on state
                if (script.has_error) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                } else if (!script.enabled) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
                }

                std::string filename = script.path.filename().string();
                ImGui::Text("%s", filename.c_str());
                ImGui::PopStyleColor();

                // Tooltip with full path and error
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Path: %s", script.path.string().c_str());
                    if (script.has_error) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", script.error_message.c_str());
                    }
                    ImGui::EndTooltip();
                }

                // Reload single script button
                ImGui::SameLine(ImGui::GetWindowWidth() - 80);
                if (ImGui::SmallButton("Reload")) {
                    state.setScriptError(i, "");
                    if (script.enabled) {
                        auto result = lfs::python::run_scripts({script.path});
                        if (!result) {
                            state.setScriptError(i, result.error());
                        }
                    }
                }

                ImGui::PopID();
            }
        }

        ImGui::End();
#endif // LFS_BUILD_PYTHON_BINDINGS
    }

} // namespace lfs::vis::gui::panels
