/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_server.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"

#include <cstring>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace lfs::vis {

    using json = nlohmann::json;
    using namespace lfs::core::events;

    SelectionServer::SelectionServer() = default;

    SelectionServer::~SelectionServer() {
        stop();
    }

    bool SelectionServer::start(const std::string& socket_path) {
        if (running_)
            return true;

        socket_path_ = socket_path;
        unlink(socket_path_.c_str());

        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            LOG_ERROR("SelectionServer: socket() failed: {}", strerror(errno));
            return false;
        }

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            LOG_ERROR("SelectionServer: bind() failed: {}", strerror(errno));
            close(server_fd_);
            server_fd_ = -1;
            return false;
        }

        if (listen(server_fd_, LISTEN_BACKLOG) < 0) {
            LOG_ERROR("SelectionServer: listen() failed: {}", strerror(errno));
            close(server_fd_);
            server_fd_ = -1;
            return false;
        }

        running_ = true;
        server_thread_ = std::thread(&SelectionServer::server_loop, this);
        LOG_INFO("SelectionServer started: {}", socket_path_);
        return true;
    }

    void SelectionServer::stop() {
        if (!running_)
            return;

        running_ = false;

        if (server_fd_ >= 0) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }

        if (server_thread_.joinable())
            server_thread_.join();

        unlink(socket_path_.c_str());
    }

    void SelectionServer::server_loop() {
        while (running_) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(server_fd_, &read_fds);

            timeval timeout{.tv_sec = 0, .tv_usec = SELECT_TIMEOUT_US};
            const int result = select(server_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);

            if (result < 0) {
                if (running_)
                    LOG_ERROR("SelectionServer: select() failed");
                break;
            }

            if (result == 0)
                continue;

            const int client_fd = accept(server_fd_, nullptr, nullptr);
            if (client_fd < 0)
                continue;

            handle_client(client_fd);
            close(client_fd);
        }
    }

    void SelectionServer::queue_command(SelectionCommand cmd) {
        std::lock_guard lock(command_queue_mutex_);
        command_queue_.push(std::move(cmd));
    }

    void SelectionServer::process_pending_commands() {
        std::queue<SelectionCommand> commands;
        {
            std::lock_guard lock(command_queue_mutex_);
            std::swap(commands, command_queue_);
        }

        while (!commands.empty()) {
            auto& cmd = commands.front();

            std::visit(
                [](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SelectRectCmd>) {
                        cmd::SelectRect{.x0 = arg.x0,
                                        .y0 = arg.y0,
                                        .x1 = arg.x1,
                                        .y1 = arg.y1,
                                        .camera_index = arg.camera_index,
                                        .mode = std::move(arg.mode)}
                            .emit();
                    } else if constexpr (std::is_same_v<T, ApplyMaskCmd>) {
                        cmd::ApplySelectionMask{.mask = std::move(arg.mask)}.emit();
                    } else if constexpr (std::is_same_v<T, DeselectAllCmd>) {
                        cmd::DeselectAll{}.emit();
                    }
                },
                cmd);

            commands.pop();
        }
    }

    void SelectionServer::send_response(int client_fd, bool success, const char* error) {
        const json response = error ? json{{"success", success}, {"error", error}} : json{{"success", success}};
        const std::string str = response.dump();
        write(client_fd, str.c_str(), str.size());
    }

    void SelectionServer::handle_client(const int client_fd) {
        char buffer[RECV_BUFFER_SIZE];
        const ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0)
            return;

        buffer[bytes_read] = '\0';

        try {
            const auto request = json::parse(buffer);
            const auto command = request.value("command", "");

            if (command == "select_rect") {
                queue_command(SelectRectCmd{.x0 = request.value("x0", 0.0f),
                                            .y0 = request.value("y0", 0.0f),
                                            .x1 = request.value("x1", 0.0f),
                                            .y1 = request.value("y1", 0.0f),
                                            .camera_index = request.value("camera_index", 0),
                                            .mode = request.value("mode", "replace")});
                send_response(client_fd, true);

            } else if (command == "apply_mask") {
                std::vector<uint8_t> mask;
                if (request.contains("mask")) {
                    mask = request["mask"].get<std::vector<uint8_t>>();
                }
                queue_command(ApplyMaskCmd{.mask = std::move(mask)});
                send_response(client_fd, true);

            } else if (command == "deselect_all") {
                queue_command(DeselectAllCmd{});
                send_response(client_fd, true);

            } else if (command == "invoke_capability") {
                const auto capability = request.value("capability", "");
                const auto args = request.value("args", "{}");
                if (capability.empty()) {
                    send_response(client_fd, false, "Missing capability name");
                } else if (!invoke_capability_callback_) {
                    LOG_ERROR("SelectionServer: capability callback not set");
                    send_response(client_fd, false, "Capability invocation not available");
                } else {
                    const auto result = invoke_capability_callback_(capability, args);
                    json response;
                    response["success"] = result.success;
                    if (!result.result_json.empty()) {
                        try {
                            response["result"] = json::parse(result.result_json);
                        } catch (...) {
                            response["result"] = result.result_json;
                        }
                    }
                    if (!result.error.empty())
                        response["error"] = result.error;
                    const std::string str = response.dump();
                    write(client_fd, str.c_str(), str.size());
                }

            } else {
                send_response(client_fd, false, "Unknown command");
            }

        } catch (const std::exception& e) {
            LOG_ERROR("SelectionServer: {}", e.what());
            send_response(client_fd, false, e.what());
        }
    }

} // namespace lfs::vis
