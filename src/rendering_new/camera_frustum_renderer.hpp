/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace lfs::rendering {

class CameraFrustumRenderer {
public:
    CameraFrustumRenderer() = default;
    ~CameraFrustumRenderer() = default;

    Result<void> init();
    Result<void> render(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                        const glm::mat4& view,
                        const glm::mat4& projection,
                        float scale = 0.1f,
                        const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
                        const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
                        const glm::mat4& scene_transform = glm::mat4(1.0f));

    Result<int> pickCamera(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                           const glm::vec2& mouse_pos,
                           const glm::vec2& viewport_pos,
                           const glm::vec2& viewport_size,
                           const glm::mat4& view,
                           const glm::mat4& projection,
                           float scale = 0.1f,
                           const glm::mat4& scene_transform = glm::mat4(1.0f));

    void setHighlightedCamera(const int index) { highlighted_camera_ = index; }
    [[nodiscard]] int getHighlightedCamera() const { return highlighted_camera_; }

    [[nodiscard]] bool isInitialized() const { return initialized_; }

    void clearCache();

private:
    struct Vertex {
        glm::vec3 position;
        glm::vec2 uv;
    };

    struct InstanceData {
        glm::mat4 transform;
        glm::vec3 color;
        float alpha;
        uint32_t is_validation;
        uint32_t padding[3];  // 16-byte alignment
    };

    Result<void> createGeometry();
    Result<void> createPickingFBO();

    void prepareInstances(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                          float scale,
                          const glm::vec3& train_color,
                          const glm::vec3& eval_color,
                          bool for_picking,
                          const glm::vec3& view_position,
                          const glm::mat4& scene_transform);

    void updateInstanceVisibility(const glm::vec3& view_position);

    // GL resources
    ManagedShader shader_;
    VAO vao_;
    VBO vbo_;
    EBO face_ebo_;
    EBO edge_ebo_;
    VBO instance_vbo_;

    // Picking
    FBO picking_fbo_;
    Texture picking_color_texture_;
    Texture picking_depth_texture_;
    int picking_fbo_width_ = 0;
    int picking_fbo_height_ = 0;

    // Instance data
    std::vector<InstanceData> cached_instances_;
    std::vector<int> camera_ids_;
    std::vector<glm::vec3> camera_positions_;

    size_t num_face_indices_ = 0;
    size_t num_edge_indices_ = 0;
    int highlighted_camera_ = -1;
    bool initialized_ = false;

    // Cache invalidation
    float last_scale_ = -1.0f;
    glm::vec3 last_train_color_{-1, -1, -1};
    glm::vec3 last_eval_color_{-1, -1, -1};
    glm::vec3 last_view_position_{0, 0, 0};
    glm::mat4 last_scene_transform_{1.0f};
};

} // namespace lfs::rendering
