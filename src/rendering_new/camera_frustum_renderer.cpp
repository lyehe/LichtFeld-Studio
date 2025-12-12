/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_frustum_renderer.hpp"
#include "core_new/logger.hpp"
#include "gl_state_guard.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::rendering {

namespace {
    constexpr float FADE_START_MULTIPLIER = 5.0f;
    constexpr float FADE_END_MULTIPLIER = 0.2f;
    constexpr float MIN_VISIBLE_MULTIPLIER = 0.1f;
    constexpr float MIN_VISIBLE_ALPHA = 0.05f;
    constexpr float MIN_RENDER_ALPHA = 0.01f;
    constexpr float WIREFRAME_WIDTH = 1.5f;
    constexpr int PICKING_SAMPLE_SIZE = 3;

    const glm::mat4 GL_TO_COLMAP = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
}

void CameraFrustumRenderer::clearCache() {
    cached_instances_.clear();
    camera_ids_.clear();
    camera_positions_.clear();
    last_scale_ = -1.0f;
}

Result<void> CameraFrustumRenderer::init() {
    auto shader_result = load_shader("camera_frustum", "camera_frustum.vert", "camera_frustum.frag", false);
    if (!shader_result) {
        return std::unexpected(shader_result.error().what());
    }
    shader_ = std::move(*shader_result);

    if (auto result = createGeometry(); !result) {
        return result;
    }

    auto instance_vbo_result = create_vbo();
    if (!instance_vbo_result) {
        return std::unexpected(instance_vbo_result.error());
    }
    instance_vbo_ = std::move(*instance_vbo_result);

    if (auto result = createPickingFBO(); !result) {
        return result;
    }

    initialized_ = true;
    LOG_INFO("Camera frustum renderer initialized");
    return {};
}

Result<void> CameraFrustumRenderer::createGeometry() {
    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, -1.0f}, {0.0f, 0.0f}},
        {{0.5f, -0.5f, -1.0f}, {1.0f, 0.0f}},
        {{0.5f, 0.5f, -1.0f}, {1.0f, 1.0f}},
        {{-0.5f, 0.5f, -1.0f}, {0.0f, 1.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f}}
    };

    const std::vector<unsigned int> face_indices = {
        0, 1, 2, 0, 2, 3,           // Base
        0, 4, 1, 1, 4, 2, 2, 4, 3, 3, 4, 0  // Sides
    };

    const std::vector<unsigned int> edge_indices = {
        0, 1, 1, 2, 2, 3, 3, 0,    // Base
        0, 4, 1, 4, 2, 4, 3, 4     // Apex
    };

    num_face_indices_ = face_indices.size();
    num_edge_indices_ = edge_indices.size();

    auto vao_result = create_vao();
    if (!vao_result) return std::unexpected(vao_result.error());

    auto vbo_result = create_vbo();
    if (!vbo_result) return std::unexpected(vbo_result.error());
    vbo_ = std::move(*vbo_result);

    auto face_ebo_result = create_vbo();
    if (!face_ebo_result) return std::unexpected(face_ebo_result.error());
    face_ebo_ = std::move(*face_ebo_result);

    auto edge_ebo_result = create_vbo();
    if (!edge_ebo_result) return std::unexpected(edge_ebo_result.error());
    edge_ebo_ = std::move(*edge_ebo_result);

    VAOBuilder builder(std::move(*vao_result));

    const std::span<const float> vertices_data(
        reinterpret_cast<const float*>(vertices.data()),
        vertices.size() * sizeof(Vertex) / sizeof(float));

    builder.attachVBO(vbo_, vertices_data, GL_STATIC_DRAW)
        .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = nullptr})
        .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = reinterpret_cast<const void*>(offsetof(Vertex, uv))});

    builder.attachEBO(face_ebo_, std::span(face_indices), GL_STATIC_DRAW);
    vao_ = builder.build();

    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
    upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(edge_indices), GL_STATIC_DRAW);

    return {};
}

Result<void> CameraFrustumRenderer::createPickingFBO() {
    GLuint fbo_id;
    glGenFramebuffers(1, &fbo_id);
    if (fbo_id == 0) {
        return std::unexpected("Failed to create picking FBO");
    }
    picking_fbo_ = FBO(fbo_id);

    picking_fbo_width_ = 256;
    picking_fbo_height_ = 256;

    GLuint color_tex;
    glGenTextures(1, &color_tex);
    picking_color_texture_ = Texture(color_tex);

    glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLuint depth_tex;
    glGenTextures(1, &depth_tex);
    picking_depth_texture_ = Texture(depth_tex);

    glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, picking_color_texture_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, picking_depth_texture_, 0);

    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        return std::unexpected("Picking FBO incomplete");
    }
    return {};
}

void CameraFrustumRenderer::prepareInstances(
    const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
    const float scale,
    const glm::vec3& train_color,
    const glm::vec3& eval_color,
    const bool for_picking,
    const glm::vec3& view_position,
    const glm::mat4& scene_transform) {

    const bool needs_regeneration =
        cached_instances_.size() != cameras.size() ||
        last_scale_ != scale ||
        last_train_color_ != train_color ||
        last_eval_color_ != eval_color ||
        last_scene_transform_ != scene_transform;

    if (!needs_regeneration && !cached_instances_.empty()) {
        updateInstanceVisibility(view_position);
        return;
    }

    cached_instances_.clear();
    cached_instances_.reserve(cameras.size());
    camera_ids_.clear();
    camera_ids_.reserve(cameras.size());
    camera_positions_.clear();
    camera_positions_.reserve(cameras.size());

    for (const auto& cam : cameras) {
        auto R_tensor = cam->R();
        auto T_tensor = cam->T();

        if (!R_tensor.is_valid() || !T_tensor.is_valid()) continue;

        if (R_tensor.device() != lfs::core::Device::CPU) R_tensor = R_tensor.cpu();
        if (T_tensor.device() != lfs::core::Device::CPU) T_tensor = T_tensor.cpu();

        glm::mat4 w2c(1.0f);
        auto R_acc = R_tensor.accessor<float, 2>();
        auto T_acc = T_tensor.accessor<float, 1>();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                w2c[j][i] = R_acc(i, j);
            }
            w2c[3][i] = T_acc(i);
        }

        const glm::mat4 transformed_c2w = scene_transform * glm::inverse(w2c);
        const glm::vec3 cam_pos = glm::vec3(transformed_c2w[3]);
        camera_positions_.push_back(cam_pos);

        const float aspect = static_cast<float>(cam->image_width()) / static_cast<float>(cam->image_height());
        const float fov_y = lfs::core::focal2fov(cam->focal_y(), cam->image_height());
        const float half_height = std::tan(fov_y * 0.5f);
        const float half_width = half_height * aspect;

        const glm::mat4 fov_scale = glm::scale(glm::mat4(1.0f), glm::vec3(half_width * 2.0f * scale, half_height * 2.0f * scale, scale));
        const glm::mat4 model = transformed_c2w * GL_TO_COLMAP * fov_scale;

        const bool is_validation = cam->image_name().find("test") != std::string::npos;
        const glm::vec3 color = is_validation ? eval_color : train_color;

        float alpha = 1.0f;
        if (!for_picking) {
            const float distance = glm::length(cam_pos - view_position);
            const float fade_start = FADE_START_MULTIPLIER * scale;
            const float fade_end = FADE_END_MULTIPLIER * scale;
            const float min_visible = MIN_VISIBLE_MULTIPLIER * scale;

            if (distance < min_visible) {
                alpha = 0.0f;
            } else if (distance < fade_end) {
                alpha = MIN_VISIBLE_ALPHA;
            } else if (distance < fade_start) {
                const float t = (distance - fade_end) / (fade_start - fade_end);
                alpha = MIN_VISIBLE_ALPHA + (1.0f - MIN_VISIBLE_ALPHA) * (t * t * (3.0f - 2.0f * t));
            }
        }

        cached_instances_.push_back({model, color, alpha, is_validation ? 1u : 0u, {0, 0, 0}});
        camera_ids_.push_back(cam->uid());
    }

    last_scale_ = scale;
    last_train_color_ = train_color;
    last_eval_color_ = eval_color;
    last_view_position_ = view_position;
    last_scene_transform_ = scene_transform;
}

void CameraFrustumRenderer::updateInstanceVisibility(const glm::vec3& view_position) {
    if (camera_positions_.size() != cached_instances_.size()) return;

    const float fade_start = FADE_START_MULTIPLIER * last_scale_;
    const float fade_end = FADE_END_MULTIPLIER * last_scale_;
    const float min_visible = MIN_VISIBLE_MULTIPLIER * last_scale_;

    for (size_t i = 0; i < camera_positions_.size(); ++i) {
        const float distance = glm::length(camera_positions_[i] - view_position);
        float alpha = 1.0f;

        if (distance < min_visible) {
            alpha = 0.0f;
        } else if (distance < fade_end) {
            alpha = MIN_VISIBLE_ALPHA;
        } else if (distance < fade_start) {
            const float t = (distance - fade_end) / (fade_start - fade_end);
            alpha = MIN_VISIBLE_ALPHA + (1.0f - MIN_VISIBLE_ALPHA) * (t * t * (3.0f - 2.0f * t));
        }
        cached_instances_[i].alpha = alpha;
    }
    last_view_position_ = view_position;
}

Result<void> CameraFrustumRenderer::render(
    const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
    const glm::mat4& view,
    const glm::mat4& projection,
    const float scale,
    const glm::vec3& train_color,
    const glm::vec3& eval_color,
    const glm::mat4& scene_transform) {

    if (!initialized_ || cameras.empty()) return {};

    const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
    prepareInstances(cameras, scale, train_color, eval_color, false, view_position, scene_transform);

    if (cached_instances_.empty()) return {};

    std::vector<InstanceData> visible_instances;
    std::vector<int> visible_indices;
    visible_instances.reserve(cached_instances_.size());
    visible_indices.reserve(cached_instances_.size());

    for (size_t i = 0; i < cached_instances_.size(); ++i) {
        if (cached_instances_[i].alpha > MIN_RENDER_ALPHA) {
            visible_instances.push_back(cached_instances_[i]);
            visible_indices.push_back(static_cast<int>(i));
        }
    }

    if (visible_instances.empty()) return {};

    GLStateGuard state_guard;
    while (glGetError() != GL_NO_ERROR) {}

    {
        ShaderScope shader(shader_);
        if (!shader.isBound()) {
            return std::unexpected("Failed to bind camera frustum shader");
        }

        const glm::mat4 view_proj = projection * view;
        shader->set("viewProj", view_proj);
        shader->set("viewPos", view_position);
        shader->set("pickingMode", false);

        int visible_highlight_index = -1;
        for (size_t i = 0; i < visible_indices.size(); ++i) {
            if (visible_indices[i] == highlighted_camera_) {
                visible_highlight_index = static_cast<int>(i);
                break;
            }
        }
        shader->set("highlightIndex", visible_highlight_index);

        {
            VAOBinder vao_bind(vao_);

            {
                BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                upload_buffer(GL_ARRAY_BUFFER, std::span(visible_instances), GL_DYNAMIC_DRAW);

                for (int i = 0; i < 4; ++i) {
                    glEnableVertexAttribArray(2 + i);
                    glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                    glVertexAttribDivisor(2 + i, 1);
                }

                glEnableVertexAttribArray(6);
                glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(offsetof(InstanceData, color)));
                glVertexAttribDivisor(6, 1);

                glEnableVertexAttribArray(7);
                glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                       reinterpret_cast<void*>(offsetof(InstanceData, is_validation)));
                glVertexAttribDivisor(7, 1);
            }

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glLineWidth(WIREFRAME_WIDTH);
            {
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                glDrawElementsInstanced(GL_LINES, static_cast<GLsizei>(num_edge_indices_), GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(visible_instances.size()));
            }

            for (int i = 2; i <= 7; ++i) {
                glDisableVertexAttribArray(i);
                glVertexAttribDivisor(i, 0);
            }
        }
    }

    glFinish();

    return {};
}

Result<int> CameraFrustumRenderer::pickCamera(
    const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
    const glm::vec2& mouse_pos,
    const glm::vec2& viewport_pos,
    const glm::vec2& viewport_size,
    const glm::mat4& view,
    const glm::mat4& projection,
    const float scale,
    const glm::mat4& scene_transform) {

    if (!initialized_ || cameras.empty()) return -1;

    if (cached_instances_.empty() || camera_ids_.size() != cameras.size()) {
        const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
        prepareInstances(cameras, scale, last_train_color_, last_eval_color_, false, view_position, scene_transform);
        if (cached_instances_.empty()) return -1;
    }

    const int vp_width = static_cast<int>(viewport_size.x);
    const int vp_height = static_cast<int>(viewport_size.y);

    if (vp_width != picking_fbo_width_ || vp_height != picking_fbo_height_) {
        picking_fbo_width_ = vp_width;
        picking_fbo_height_ = vp_height;

        glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_, 0, GL_RGB, GL_FLOAT, nullptr);

        glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }

    GLint current_fbo;
    GLint current_viewport[4];
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
    glGetIntegerv(GL_VIEWPORT, current_viewport);

    glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
    glViewport(0, 0, picking_fbo_width_, picking_fbo_height_);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
        ShaderScope shader(shader_);
        if (!shader.isBound()) {
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
            return std::unexpected("Failed to bind picking shader");
        }

        const glm::mat4 view_proj = projection * view;
        const glm::vec3 view_pos = glm::vec3(glm::inverse(view)[3]);

        shader->set("viewProj", view_proj);
        shader->set("viewPos", view_pos);
        shader->set("pickingMode", true);
        shader->set("minimumPickDistance", scale * 2.0f);

        VAOBinder vao_bind(vao_);

        {
            BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
            upload_buffer(GL_ARRAY_BUFFER, std::span(cached_instances_), GL_DYNAMIC_DRAW);

            for (int i = 0; i < 4; ++i) {
                glEnableVertexAttribArray(2 + i);
                glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                glVertexAttribDivisor(2 + i, 1);
            }

            glEnableVertexAttribArray(6);
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                  reinterpret_cast<void*>(offsetof(InstanceData, color)));
            glVertexAttribDivisor(6, 1);

            glEnableVertexAttribArray(7);
            glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_validation)));
            glVertexAttribDivisor(7, 1);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        {
            BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
            glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(num_face_indices_), GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(cached_instances_.size()));
        }

        for (int i = 2; i <= 7; ++i) {
            glDisableVertexAttribArray(i);
            glVertexAttribDivisor(i, 0);
        }
    }

    glFinish();

    const int pixel_x = std::clamp(static_cast<int>(mouse_pos.x - viewport_pos.x), 0, picking_fbo_width_ - 1);
    const int pixel_y = std::clamp(static_cast<int>(viewport_size.y - (mouse_pos.y - viewport_pos.y)), 0, picking_fbo_height_ - 1);

    const int read_x = std::max(0, pixel_x - 1);
    const int read_y = std::max(0, pixel_y - 1);
    const int read_width = std::min(PICKING_SAMPLE_SIZE, picking_fbo_width_ - read_x);
    const int read_height = std::min(PICKING_SAMPLE_SIZE, picking_fbo_height_ - read_y);

    std::vector<float> pixels(read_width * read_height * 3);
    glReadPixels(read_x, read_y, read_width, read_height, GL_RGB, GL_FLOAT, pixels.data());

    int center_idx = 0;
    if (read_width == 3 && read_height == 3) {
        center_idx = 4 * 3;
    } else if (read_width >= 2 && read_height >= 2) {
        center_idx = ((read_height / 2) * read_width + (read_width / 2)) * 3;
    }

    const int id = (static_cast<int>(pixels[center_idx] * 255.0f + 0.5f) << 16 |
                    static_cast<int>(pixels[center_idx + 1] * 255.0f + 0.5f) << 8 |
                    static_cast<int>(pixels[center_idx + 2] * 255.0f + 0.5f)) - 1;

    glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
    glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

    if (id >= 0 && id < static_cast<int>(camera_ids_.size())) {
        return camera_ids_[id];
    }
    return -1;
}

} // namespace lfs::rendering
