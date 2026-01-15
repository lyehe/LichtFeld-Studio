/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef LFS_PANEL_REGISTRY_EXPORTS
#define LFS_PANEL_REGISTRY_API __declspec(dllexport)
#else
#define LFS_PANEL_REGISTRY_API __declspec(dllimport)
#endif
#else
#define LFS_PANEL_REGISTRY_API
#endif

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
    using EnsureInitializedCallback = std::function<void()>;

    // Register callbacks from the Python module
    LFS_PANEL_REGISTRY_API void set_ensure_initialized_callback(EnsureInitializedCallback cb);
    LFS_PANEL_REGISTRY_API void set_panel_draw_callback(DrawPanelsCallback cb);
    LFS_PANEL_REGISTRY_API void set_panel_draw_single_callback(DrawSinglePanelCallback cb);
    LFS_PANEL_REGISTRY_API void set_panel_has_callback(HasPanelsCallback cb);
    LFS_PANEL_REGISTRY_API void set_panel_names_callback(GetPanelNamesCallback cb);
    LFS_PANEL_REGISTRY_API void set_python_cleanup_callback(CleanupCallback cb);
    LFS_PANEL_REGISTRY_API void clear_panel_callbacks();

    // C++ interface for the visualizer
    LFS_PANEL_REGISTRY_API void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    LFS_PANEL_REGISTRY_API void draw_python_panel(const std::string& name, lfs::vis::Scene* scene = nullptr);
    LFS_PANEL_REGISTRY_API bool has_python_panels(PanelSpace space);
    LFS_PANEL_REGISTRY_API std::vector<std::string> get_python_panel_names(PanelSpace space);
    LFS_PANEL_REGISTRY_API void invoke_python_cleanup();

    // Operation context for Python code (short-lived, per-call)
    LFS_PANEL_REGISTRY_API void set_scene_for_python(void* scene);
    LFS_PANEL_REGISTRY_API void* get_scene_for_python();

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
    class LFS_PANEL_REGISTRY_API ApplicationSceneContext {
    public:
        void set(vis::Scene* scene);
        vis::Scene* get() const;
        uint64_t generation() const;

    private:
        std::atomic<vis::Scene*> scene_{nullptr};
        std::atomic<uint64_t> generation_{0};
    };

    LFS_PANEL_REGISTRY_API void set_application_scene(vis::Scene* scene);
    LFS_PANEL_REGISTRY_API vis::Scene* get_application_scene();
    LFS_PANEL_REGISTRY_API uint64_t get_scene_generation();

} // namespace lfs::python
