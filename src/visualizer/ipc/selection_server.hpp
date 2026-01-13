/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace lfs::vis {

struct SelectRectCmd {
    float x0, y0, x1, y1;
    int camera_index;
    std::string mode;
};

struct ApplyMaskCmd {
    std::vector<uint8_t> mask;
};

struct DeselectAllCmd {};

using SelectionCommand = std::variant<SelectRectCmd, ApplyMaskCmd, DeselectAllCmd>;

struct CapabilityInvokeResult {
    bool success = false;
    std::string result_json;
    std::string error;
};

using InvokeCapabilityCallback = std::function<CapabilityInvokeResult(const std::string& name, const std::string& args)>;

class SelectionServer {
public:
    static constexpr const char* SOCKET_PATH = "/tmp/lichtfeld-selection.sock";

    SelectionServer();
    ~SelectionServer();

    SelectionServer(const SelectionServer&) = delete;
    SelectionServer& operator=(const SelectionServer&) = delete;

    bool start(const std::string& socket_path = SOCKET_PATH);
    void stop();
    [[nodiscard]] bool is_running() const { return running_; }

    void process_pending_commands();

    void setInvokeCapabilityCallback(InvokeCapabilityCallback callback) { invoke_capability_callback_ = std::move(callback); }

private:
    static constexpr size_t RECV_BUFFER_SIZE = 65536;
    static constexpr int LISTEN_BACKLOG = 5;
    static constexpr int SELECT_TIMEOUT_US = 100000;

    void server_loop();
    void handle_client(int client_fd);
    void queue_command(SelectionCommand cmd);
    void send_response(int client_fd, bool success, const char* error = nullptr);

    std::string socket_path_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread server_thread_;

    std::mutex command_queue_mutex_;
    std::queue<SelectionCommand> command_queue_;

    InvokeCapabilityCallback invoke_capability_callback_;
};

} // namespace lfs::vis
