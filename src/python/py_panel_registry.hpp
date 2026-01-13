/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lfs::vis {
    class Scene;
}

namespace lfs::python {

    // Panel space types
    enum class PanelSpace {
        SidePanel,
        Floating,
        ViewportOverlay
    };

    // Callback types for the Python panel system
    using DrawPanelsCallback = std::function<void(PanelSpace)>;
    using DrawSinglePanelCallback = std::function<void(const std::string&)>;
    using HasPanelsCallback = std::function<bool(PanelSpace)>;
    using GetPanelNamesCallback = std::function<std::vector<std::string>(PanelSpace)>;
    using CleanupCallback = std::function<void()>;

    // Register callbacks from the Python module
    void set_panel_draw_callback(DrawPanelsCallback cb);
    void set_panel_draw_single_callback(DrawSinglePanelCallback cb);
    void set_panel_has_callback(HasPanelsCallback cb);
    void set_panel_names_callback(GetPanelNamesCallback cb);
    void set_python_cleanup_callback(CleanupCallback cb);
    void clear_panel_callbacks();

    // C++ interface for the visualizer
    void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    void draw_python_panel(const std::string& name, lfs::vis::Scene* scene = nullptr);
    bool has_python_panels(PanelSpace space);
    std::vector<std::string> get_python_panel_names(PanelSpace space);
    void invoke_python_cleanup();

    // Operation context for Python code (short-lived, per-call)
    void set_scene_for_python(void* scene);
    void* get_scene_for_python();

    // RAII guard for operation context (used for capability invocations)
    class SceneContextGuard {
    public:
        explicit SceneContextGuard(void* scene) { set_scene_for_python(scene); }
        ~SceneContextGuard() { set_scene_for_python(nullptr); }

        SceneContextGuard(const SceneContextGuard&) = delete;
        SceneContextGuard& operator=(const SceneContextGuard&) = delete;
        SceneContextGuard(SceneContextGuard&&) = delete;
        SceneContextGuard& operator=(SceneContextGuard&&) = delete;
    };

    // Application scene context (long-lived, persists while model is loaded)
    class ApplicationSceneContext {
    public:
        void set(vis::Scene* scene);
        vis::Scene* get() const;
        uint64_t generation() const;

    private:
        std::atomic<vis::Scene*> scene_{nullptr};
        std::atomic<uint64_t> generation_{0};
    };

    void set_application_scene(vis::Scene* scene);
    vis::Scene* get_application_scene();

} // namespace lfs::python
