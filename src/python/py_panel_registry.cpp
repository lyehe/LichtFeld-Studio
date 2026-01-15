/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_panel_registry.hpp"

namespace lfs::python {

    namespace {
        EnsureInitializedCallback g_ensure_initialized_callback;
        DrawPanelsCallback g_draw_callback;
        DrawSinglePanelCallback g_draw_single_callback;
        HasPanelsCallback g_has_callback;
        GetPanelNamesCallback g_panel_names_callback;
        CleanupCallback g_cleanup_callback;

        // Operation context (short-lived, per-call)
        void* g_scene_for_python = nullptr;

        // Application context (long-lived, persists while model is loaded)
        ApplicationSceneContext g_app_scene_context;
    } // namespace

    // Operation context (short-lived)
    void set_scene_for_python(void* scene) { g_scene_for_python = scene; }
    void* get_scene_for_python() { return g_scene_for_python; }

    // Application context (long-lived)
    void ApplicationSceneContext::set(vis::Scene* scene) {
        scene_.store(scene);
        generation_.fetch_add(1);
    }

    vis::Scene* ApplicationSceneContext::get() const { return scene_.load(); }

    uint64_t ApplicationSceneContext::generation() const { return generation_.load(); }

    void set_application_scene(vis::Scene* scene) { g_app_scene_context.set(scene); }

    vis::Scene* get_application_scene() { return g_app_scene_context.get(); }

    uint64_t get_scene_generation() { return g_app_scene_context.generation(); }

    void set_ensure_initialized_callback(EnsureInitializedCallback cb) {
        g_ensure_initialized_callback = std::move(cb);
    }

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        g_draw_callback = std::move(cb);
    }

    void set_panel_draw_single_callback(DrawSinglePanelCallback cb) {
        g_draw_single_callback = std::move(cb);
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        g_has_callback = std::move(cb);
    }

    void set_panel_names_callback(GetPanelNamesCallback cb) {
        g_panel_names_callback = std::move(cb);
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        g_cleanup_callback = std::move(cb);
    }

    void clear_panel_callbacks() {
        g_ensure_initialized_callback = nullptr;
        g_draw_callback = nullptr;
        g_draw_single_callback = nullptr;
        g_has_callback = nullptr;
        g_panel_names_callback = nullptr;
        g_cleanup_callback = nullptr;
    }

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space, lfs::vis::Scene* /* scene */) {
        if (!g_draw_callback)
            return;
        // No guard needed - panels use persistent application scene context
        g_draw_callback(space);
    }

    bool has_python_panels(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (g_has_callback) {
            return g_has_callback(space);
        }
        return false;
    }

    std::vector<std::string> get_python_panel_names(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (g_panel_names_callback) {
            return g_panel_names_callback(space);
        }
        return {};
    }

    void draw_python_panel(const std::string& name, lfs::vis::Scene* /* scene */) {
        if (!g_draw_single_callback)
            return;
        // No guard needed - panels use persistent application scene context
        g_draw_single_callback(name);
    }

} // namespace lfs::python
