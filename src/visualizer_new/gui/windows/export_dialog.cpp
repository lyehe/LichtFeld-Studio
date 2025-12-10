/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/export_dialog.hpp"
#include "scene/scene_manager.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    using ExportFormat = lfs::core::ExportFormat;

    void ExportDialog::render(bool* p_open, SceneManager* scene_manager) {
        if (!p_open || !*p_open) return;

        constexpr ImGuiWindowFlags WINDOW_FLAGS =
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;

        ImGui::SetNextWindowSize(ImVec2(400, 0), ImGuiCond_FirstUseEver);

        if (!ImGui::Begin("Export", p_open, WINDOW_FLAGS)) {
            ImGui::End();
            return;
        }

        // Collect SPLAT nodes from scene
        std::vector<const SceneNode*> splat_nodes;
        if (scene_manager) {
            const auto& scene = scene_manager->getScene();
            for (const auto* node : scene.getNodes()) {
                if (node->type == NodeType::SPLAT && node->model) {
                    splat_nodes.push_back(node);
                }
            }
        }

        // Initialize on first open
        if (!initialized_ && !splat_nodes.empty()) {
            selected_nodes_.clear();
            for (const auto* node : splat_nodes) {
                selected_nodes_.insert(node->name);
            }
            initialized_ = true;
        }

        // === Format Selection ===
        ImGui::Text("Export Format");
        ImGui::Separator();

        int format_idx = static_cast<int>(selected_format_);
        ImGui::RadioButton("PLY (Standard)", &format_idx, static_cast<int>(ExportFormat::PLY));
        ImGui::RadioButton("Compressed PLY", &format_idx, static_cast<int>(ExportFormat::COMPRESSED_PLY));
        ImGui::RadioButton("SOG (SuperSplat)", &format_idx, static_cast<int>(ExportFormat::SOG));
        ImGui::RadioButton("HTML Viewer", &format_idx, static_cast<int>(ExportFormat::HTML_VIEWER));
        selected_format_ = static_cast<ExportFormat>(format_idx);

        ImGui::Spacing();
        ImGui::Spacing();

        // === Node Selection ===
        ImGui::Text("Select Models to Export");
        ImGui::Separator();

        if (splat_nodes.empty()) {
            ImGui::TextDisabled("No models in scene");
        } else {
            if (ImGui::SmallButton("All")) {
                for (const auto* node : splat_nodes) {
                    selected_nodes_.insert(node->name);
                }
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("None")) {
                selected_nodes_.clear();
            }

            ImGui::Spacing();

            for (const auto* node : splat_nodes) {
                bool selected = selected_nodes_.contains(node->name);
                if (ImGui::Checkbox(node->name.c_str(), &selected)) {
                    if (selected) {
                        selected_nodes_.insert(node->name);
                    } else {
                        selected_nodes_.erase(node->name);
                    }
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(%zu gaussians)", node->gaussian_count);
            }
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();

        // === Export Button ===
        const bool can_export = !selected_nodes_.empty();

        if (!can_export) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Select at least one model");
        }

        ImGui::BeginDisabled(!can_export);

        const char* export_label = selected_nodes_.size() > 1 ? "Export Merged..." : "Export...";
        if (ImGui::Button(export_label, ImVec2(140, 0))) {
            if (on_browse_) {
                // Generate default filename
                std::string default_name = "export";
                if (selected_nodes_.size() == 1) {
                    default_name = *selected_nodes_.begin();
                } else {
                    default_name = "merged";
                }

                std::vector<std::string> nodes(selected_nodes_.begin(), selected_nodes_.end());
                on_browse_(selected_format_, default_name, nodes);
            }
            *p_open = false;
            initialized_ = false;
        }

        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            *p_open = false;
            initialized_ = false;
        }

        ImGui::End();
    }

} // namespace lfs::vis::gui
