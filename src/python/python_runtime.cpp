/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#endif

namespace lfs::python {

    namespace {
        EnsureInitializedCallback g_ensure_initialized_callback = nullptr;
        DrawPanelsCallback g_draw_callback = nullptr;
        DrawSinglePanelCallback g_draw_single_callback = nullptr;
        HasPanelsCallback g_has_callback = nullptr;
        GetPanelNamesCallback g_panel_names_callback = nullptr;
        CleanupCallback g_cleanup_callback = nullptr;

        // Unique ID for this DLL instance (detects duplicate loading)
        const void* const DLL_INSTANCE_ID = &g_draw_single_callback;

        // Operation context (short-lived, per-call)
        void* g_scene_for_python = nullptr;

        ApplicationSceneContext g_app_scene_context;
        std::atomic<bool> g_gil_state_ready{false};

#ifdef LFS_BUILD_PYTHON_BINDINGS
        // Main thread GIL state - stored here in the shared library
        // to ensure single copy across all modules
        std::atomic<PyThreadState*> g_main_thread_state{nullptr};
#endif

        // Initialization guards - must be in shared library
        std::once_flag g_py_init_once;
        std::once_flag g_redirect_once;
        std::atomic<bool> g_plugins_loaded{false};
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

    void set_gil_state_ready(const bool ready) { g_gil_state_ready.store(ready, std::memory_order_release); }
    bool is_gil_state_ready() { return g_gil_state_ready.load(std::memory_order_acquire); }

#ifdef LFS_BUILD_PYTHON_BINDINGS
    void set_main_thread_state(void* state) {
        g_main_thread_state.store(static_cast<PyThreadState*>(state), std::memory_order_release);
    }

    void* get_main_thread_state() {
        return g_main_thread_state.load(std::memory_order_acquire);
    }

    void acquire_gil_main_thread() {
        PyThreadState* state = g_main_thread_state.exchange(nullptr, std::memory_order_acq_rel);
        if (state) {
            PyEval_RestoreThread(state);
        }
    }

    void release_gil_main_thread() {
        PyThreadState* state = PyEval_SaveThread();
        g_main_thread_state.store(state, std::memory_order_release);
    }
#else
    // Stubs when Python bindings disabled
    void set_main_thread_state(void*) {}
    void* get_main_thread_state() { return nullptr; }
    void acquire_gil_main_thread() {}
    void release_gil_main_thread() {}
#endif

    // Initialization guards
    void call_once_py_init(std::function<void()> fn) {
        std::call_once(g_py_init_once, std::move(fn));
    }

    void call_once_redirect(std::function<void()> fn) {
        std::call_once(g_redirect_once, std::move(fn));
    }

    void mark_plugins_loaded() { g_plugins_loaded.store(true, std::memory_order_release); }
    bool are_plugins_loaded() { return g_plugins_loaded.load(std::memory_order_acquire); }

    void set_ensure_initialized_callback(EnsureInitializedCallback cb) {
        std::fprintf(stderr, "[pyrt] set_ensure_initialized_callback: %p -> %p (globals at %p)\n",
                     reinterpret_cast<void*>(g_ensure_initialized_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_ensure_initialized_callback));
        std::fflush(stderr);
        g_ensure_initialized_callback = cb;
    }

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_draw_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_draw_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_draw_callback));
        std::fflush(stderr);
        g_draw_callback = cb;
    }

    void set_panel_draw_single_callback(DrawSinglePanelCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_draw_single_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_draw_single_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_draw_single_callback));
        std::fflush(stderr);
        g_draw_single_callback = cb;
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_has_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_has_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_has_callback));
        std::fflush(stderr);
        g_has_callback = cb;
    }

    void set_panel_names_callback(GetPanelNamesCallback cb) {
        std::fprintf(stderr, "[pyrt] set_panel_names_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_panel_names_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_panel_names_callback));
        std::fflush(stderr);
        g_panel_names_callback = cb;
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        std::fprintf(stderr, "[pyrt] set_python_cleanup_callback: %p -> %p (global at %p)\n",
                     reinterpret_cast<void*>(g_cleanup_callback),
                     reinterpret_cast<void*>(cb),
                     reinterpret_cast<void*>(&g_cleanup_callback));
        std::fflush(stderr);
        g_cleanup_callback = cb;
    }

    void clear_panel_callbacks() {
        g_ensure_initialized_callback = nullptr;
        g_draw_callback = nullptr;
        g_draw_single_callback = nullptr;
        g_has_callback = nullptr;
        g_panel_names_callback = nullptr;
        g_cleanup_callback = nullptr;
    }

    void debug_dump_callbacks(const char* caller) {
        std::fprintf(stderr, "\n[pyrt] ======== CALLBACK STATE DUMP ========\n");
        std::fprintf(stderr, "[pyrt] Caller: %s\n", caller);
        std::fprintf(stderr, "[pyrt] DLL_INSTANCE_ID: %p  <-- MUST MATCH between pyd and exe!\n", DLL_INSTANCE_ID);
        std::fprintf(stderr, "[pyrt] g_draw_single_callback:\n");
        std::fprintf(stderr, "[pyrt]   value = %p %s\n",
                     reinterpret_cast<void*>(g_draw_single_callback),
                     g_draw_single_callback ? "(SET)" : "(NULL - PROBLEM!)");
        std::fprintf(stderr, "[pyrt]   &addr = %p\n", reinterpret_cast<void*>(&g_draw_single_callback));
        std::fprintf(stderr, "[pyrt] g_draw_callback:        value=%p\n", reinterpret_cast<void*>(g_draw_callback));
        std::fprintf(stderr, "[pyrt] g_has_callback:         value=%p\n", reinterpret_cast<void*>(g_has_callback));
        std::fprintf(stderr, "[pyrt] g_panel_names_callback: value=%p\n", reinterpret_cast<void*>(g_panel_names_callback));
        std::fprintf(stderr, "[pyrt] ======== END DUMP ========\n\n");
        std::fflush(stderr);
    }

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space, lfs::vis::Scene* /* scene */) {
        if (!g_draw_callback)
            return;
#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_draw_callback(space);
        PyGILState_Release(gil);
#else
        g_draw_callback(space);
#endif
    }

    bool has_python_panels(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (g_has_callback) {
#ifdef LFS_BUILD_PYTHON_BINDINGS
            if (!Py_IsInitialized() || !is_gil_state_ready())
                return false;
            const PyGILState_STATE gil = PyGILState_Ensure();
            const bool result = g_has_callback(space);
            PyGILState_Release(gil);
            return result;
#else
            return g_has_callback(space);
#endif
        }
        return false;
    }

    std::vector<std::string> get_python_panel_names(PanelSpace space) {
        if (g_ensure_initialized_callback) {
            g_ensure_initialized_callback();
        }

        if (!g_panel_names_callback)
            return {};

#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return {};
        const PyGILState_STATE gil = PyGILState_Ensure();
#endif

        std::vector<std::string> result;
        g_panel_names_callback(
            space,
            [](const char* name, void* ctx) {
                static_cast<std::vector<std::string>*>(ctx)->emplace_back(name);
            },
            &result);

#ifdef LFS_BUILD_PYTHON_BINDINGS
        PyGILState_Release(gil);
#endif
        return result;
    }

    void draw_python_panel(const std::string& name, lfs::vis::Scene* /* scene */) {
        // First call: dump full state to detect DLL mismatch
        static bool first_call = true;
        if (first_call) {
            debug_dump_callbacks("draw_python_panel (FIRST CALL from exe)");
            first_call = false;
        }

        std::fprintf(stderr, "[pyrt] draw_python_panel('%s'): DLL_ID=%p, callback=%p\n",
                     name.c_str(), DLL_INSTANCE_ID,
                     reinterpret_cast<void*>(g_draw_single_callback));
        std::fflush(stderr);

        if (!g_draw_single_callback) {
            std::fprintf(stderr, "[pyrt] ERROR: g_draw_single_callback is NULL!\n");
            std::fprintf(stderr, "[pyrt] This means exe and pyd loaded DIFFERENT copies of lfs_python_runtime.dll\n");
            std::fprintf(stderr, "[pyrt] Check that both load from the same path.\n");
            std::fflush(stderr);
            return;
        }

#ifdef LFS_BUILD_PYTHON_BINDINGS
        if (!Py_IsInitialized() || !is_gil_state_ready()) {
            std::fprintf(stderr, "[pyrt] Python not ready: init=%d, gil=%d\n",
                         Py_IsInitialized(), is_gil_state_ready());
            std::fflush(stderr);
            return;
        }

        const PyGILState_STATE gil = PyGILState_Ensure();
        std::fprintf(stderr, "[pyrt] Calling callback %p('%s')...\n",
                     reinterpret_cast<void*>(g_draw_single_callback), name.c_str());
        std::fflush(stderr);

        g_draw_single_callback(name.c_str());

        std::fprintf(stderr, "[pyrt] Callback returned OK\n");
        std::fflush(stderr);
        PyGILState_Release(gil);
#else
        g_draw_single_callback(name.c_str());
#endif
    }

} // namespace lfs::python
