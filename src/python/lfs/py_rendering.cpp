/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rendering.hpp"
#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "training/dataset.hpp"
#include "visualizer/scene/scene.hpp"

#include <cmath>

namespace nb = nanobind;

namespace lfs::python {

static vis::Scene* g_render_scene = nullptr;

void set_render_scene_context(vis::Scene* scene) {
    g_render_scene = scene;
}

vis::Scene* get_render_scene() {
    return g_render_scene;
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

std::optional<PyTensor> render_from_camera(int camera_index, int width, int height) {
    auto* scene = get_render_scene();
    auto* model = get_model(scene);
    if (!model)
        return std::nullopt;

    const auto dataset = scene->getTrainCameras();
    if (!dataset || camera_index < 0 || static_cast<size_t>(camera_index) >= dataset->size())
        return std::nullopt;

    const auto example = dataset->get(camera_index);
    auto* cam = example.data.camera;

    const int render_width = (width > 0) ? width : cam->image_width();
    const int render_height = (height > 0) ? height : cam->image_height();

    std::unique_ptr<core::Camera> scaled_cam;
    core::Camera* render_cam = cam;

    if (render_width != cam->image_width() || render_height != cam->image_height()) {
        const float scale_x = static_cast<float>(render_width) / cam->camera_width();
        const float scale_y = static_cast<float>(render_height) / cam->camera_height();

        scaled_cam = std::make_unique<core::Camera>(
            cam->R().clone(), cam->T().clone(), cam->focal_x() * scale_x, cam->focal_y() * scale_y,
            cam->center_x() * scale_x, cam->center_y() * scale_y, cam->radial_distortion().clone(),
            cam->tangential_distortion().clone(), cam->camera_model_type(), cam->image_name(), cam->image_path(),
            cam->mask_path(), render_width, render_height, cam->uid());
        scaled_cam->set_image_dimensions(render_width, render_height);
        render_cam = scaled_cam.get();
    }

    const auto bg = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);
    auto [image, alpha] = rendering::rasterize_tensor(*render_cam, *model, bg);
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

std::optional<PyTensor> compute_screen_positions_from_camera(int camera_index) {
    auto* scene = get_render_scene();
    auto* model = get_model(scene);
    if (!model)
        return std::nullopt;

    const auto dataset = scene->getTrainCameras();
    if (!dataset || camera_index < 0 || static_cast<size_t>(camera_index) >= dataset->size())
        return std::nullopt;

    const auto example = dataset->get(camera_index);
    const auto bg = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

    core::Tensor screen_positions;
    rendering::rasterize_tensor(*example.data.camera, *model, bg, false, DEFAULT_SCALE_THRESHOLD, nullptr, nullptr,
                                nullptr, &screen_positions);

    return PyTensor(std::move(screen_positions), true);
}

std::optional<PyViewInfo> get_current_view() {
    return std::nullopt;
}

void register_rendering(nb::module_& m) {
    nb::class_<PyViewInfo>(m, "ViewInfo")
        .def_ro("rotation", &PyViewInfo::rotation)
        .def_ro("translation", &PyViewInfo::translation)
        .def_ro("width", &PyViewInfo::width)
        .def_ro("height", &PyViewInfo::height)
        .def_ro("fov_x", &PyViewInfo::fov_x)
        .def_ro("fov_y", &PyViewInfo::fov_y);

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

    m.def("render_from_camera", &render_from_camera, nb::arg("camera_index"), nb::arg("width") = -1,
          nb::arg("height") = -1,
          R"doc(
Render scene from a dataset camera.

Args:
    camera_index: Index into training camera dataset
    width: Render width (-1 for camera's native resolution)
    height: Render height (-1 for camera's native resolution)

Returns:
    Tensor [H, W, 3] RGB image on CUDA, or None if unavailable
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

    m.def("compute_screen_positions_from_camera", &compute_screen_positions_from_camera, nb::arg("camera_index"),
          R"doc(
Compute screen positions of all Gaussians from a dataset camera.

Args:
    camera_index: Index into training camera dataset

Returns:
    Tensor [N, 2] with (x, y) pixel coordinates for each Gaussian
)doc");

    m.def("get_current_view", &get_current_view, "Get current viewport camera info (None if not available)");
}

} // namespace lfs::python
