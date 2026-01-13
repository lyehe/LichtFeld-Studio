/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rendering.hpp"
#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "py_scene.hpp"
#include "python/py_panel_registry.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "training/dataset.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/scene/scene.hpp"

#include <cmath>
#include <cstring>

namespace nb = nanobind;

namespace lfs::python {

    void set_render_scene_context(vis::Scene* scene) {
        set_scene_for_python(scene);
    }

    vis::Scene* get_render_scene() {
        if (auto* app_scene = get_application_scene()) {
            return app_scene;
        }
        return static_cast<vis::Scene*>(get_scene_for_python());
    }

} // namespace lfs::python

namespace {

    constexpr float DEFAULT_FOV = 60.0f;
    constexpr float DEFAULT_SCALE_THRESHOLD = 0.01f;

    float fov_to_focal(float fov_degrees, int pixels) {
        return static_cast<float>(pixels) / (2.0f * std::tan(fov_degrees * M_PI / 360.0f));
    }

    std::unique_ptr<lfs::core::Camera> create_camera(const lfs::core::Tensor& R, const lfs::core::Tensor& T, int width,
                                                     int height, float fov_degrees) {
        const float focal = fov_to_focal(fov_degrees, width);
        const float cx = static_cast<float>(width) / 2.0f;
        const float cy = static_cast<float>(height) / 2.0f;

        auto radial = lfs::core::Tensor::zeros({6}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto tangential = lfs::core::Tensor::zeros({2}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        auto camera =
            std::make_unique<lfs::core::Camera>(R.clone(), T.clone(), focal, focal, cx, cy, std::move(radial),
                                                std::move(tangential), lfs::core::CameraModelType::PINHOLE,
                                                "virtual_camera", "", "", width, height, -1);
        camera->set_image_dimensions(width, height);
        return camera;
    }

    lfs::core::SplatData* get_model(lfs::vis::Scene* scene) {
        return scene ? const_cast<lfs::core::SplatData*>(scene->getCombinedModel()) : nullptr;
    }

} // namespace

namespace lfs::python {

    std::optional<PyTensor> render_view(const PyTensor& rotation, const PyTensor& translation, int width, int height,
                                        float fov_degrees, const PyTensor* bg_color) {
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera(rotation.tensor(), translation.tensor(), width, height, fov_degrees);

        const auto bg = bg_color ? bg_color->tensor().clone()
                                 : core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);
        return PyTensor(image.permute({1, 2, 0}), true);
    }

    std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation, int width,
                                                     int height, float fov_degrees) {
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera(rotation.tensor(), translation.tensor(), width, height, fov_degrees);
        const auto bg = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        core::Tensor screen_positions;
        rendering::rasterize_tensor(*camera, *model, bg, false, DEFAULT_SCALE_THRESHOLD, nullptr, nullptr, nullptr,
                                    &screen_positions);

        return PyTensor(std::move(screen_positions), true);
    }

    std::optional<PyViewInfo> get_current_view() {
        const auto view_info = vis::get_current_view_info();
        if (!view_info)
            return std::nullopt;

        auto R = core::Tensor::empty({3, 3}, core::Device::CPU, core::DataType::Float32);
        auto T = core::Tensor::empty({3}, core::Device::CPU, core::DataType::Float32);

        std::memcpy(R.data_ptr(), view_info->rotation.data(), 9 * sizeof(float));
        std::memcpy(T.data_ptr(), view_info->translation.data(), 3 * sizeof(float));

        return PyViewInfo{
            .rotation = PyTensor(R.cuda(), true),
            .translation = PyTensor(T.cuda(), true),
            .width = view_info->width,
            .height = view_info->height,
            .fov_x = view_info->fov,
            .fov_y = view_info->fov,
        };
    }

    std::optional<PyViewportRender> get_viewport_render() {
        const auto render = vis::get_viewport_render();
        if (!render || !render->image)
            return std::nullopt;

        // Image is [3, H, W], permute to [H, W, 3] for Python
        auto image = render->image->permute({1, 2, 0});

        std::optional<PyTensor> screen_pos;
        if (render->screen_positions) {
            screen_pos = PyTensor(*render->screen_positions, true);
        }

        return PyViewportRender{
            .image = PyTensor(std::move(image), true),
            .screen_positions = std::move(screen_pos),
        };
    }

    std::optional<PyViewportRender> capture_viewport() {
        const auto render = vis::get_viewport_render();
        if (!render || !render->image)
            return std::nullopt;

        // Clone tensors for safe async use (independent of render loop)
        auto image = render->image->clone().permute({1, 2, 0});

        std::optional<PyTensor> screen_pos;
        if (render->screen_positions) {
            screen_pos = PyTensor(render->screen_positions->clone(), true);
        }

        return PyViewportRender{
            .image = PyTensor(std::move(image), true),
            .screen_positions = std::move(screen_pos),
        };
    }

    void register_rendering(nb::module_& m) {
        nb::class_<PyViewInfo>(m, "ViewInfo")
            .def_ro("rotation", &PyViewInfo::rotation)
            .def_ro("translation", &PyViewInfo::translation)
            .def_ro("width", &PyViewInfo::width)
            .def_ro("height", &PyViewInfo::height)
            .def_ro("fov_x", &PyViewInfo::fov_x)
            .def_ro("fov_y", &PyViewInfo::fov_y);

        nb::class_<PyViewportRender>(m, "ViewportRender")
            .def_ro("image", &PyViewportRender::image)
            .def_ro("screen_positions", &PyViewportRender::screen_positions);

        m.def("get_viewport_render", &get_viewport_render,
              "Get the current viewport's rendered image and screen positions (None if not available)");

        m.def("capture_viewport", &capture_viewport,
              "Capture viewport render for async processing (clones data, safe to use from background threads)");

        m.def("render_view", &render_view, nb::arg("rotation"), nb::arg("translation"), nb::arg("width"), nb::arg("height"),
              nb::arg("fov") = DEFAULT_FOV, nb::arg("bg_color") = nb::none(),
              R"doc(
Render scene from arbitrary camera parameters.

Args:
    rotation: [3, 3] camera rotation matrix
    translation: [3] camera position
    width: Render width in pixels
    height: Render height in pixels
    fov: Field of view in degrees (default: 60)
    bg_color: Optional [3] RGB background color

Returns:
    Tensor [H, W, 3] RGB image on CUDA, or None if scene not available
)doc");

        m.def("compute_screen_positions", &compute_screen_positions, nb::arg("rotation"), nb::arg("translation"),
              nb::arg("width"), nb::arg("height"), nb::arg("fov") = DEFAULT_FOV,
              R"doc(
Compute screen positions of all Gaussians for a given camera view.

Args:
    rotation: [3, 3] camera rotation matrix
    translation: [3] camera position
    width: Viewport width in pixels
    height: Viewport height in pixels
    fov: Field of view in degrees (default: 60)

Returns:
    Tensor [N, 2] with (x, y) pixel coordinates for each Gaussian
)doc");

        m.def("get_current_view", &get_current_view, "Get current viewport camera info (None if not available)");

        m.def(
            "get_render_scene", []() -> std::optional<PyScene> {
                auto* scene = get_render_scene();
                if (!scene)
                    return std::nullopt;
                return PyScene(scene);
            },
            "Get the current render scene (None if not available)");
    }

} // namespace lfs::python
