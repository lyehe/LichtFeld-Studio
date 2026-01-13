/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_client.hpp"

#include <cstring>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace lfs::mcp {

using json = nlohmann::json;

namespace {

std::expected<void, std::string> parse_response(const std::string& response) {
    try {
        const auto j = json::parse(response);
        if (!j.value("success", false)) {
            return std::unexpected(j.value("error", "Unknown error"));
        }
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(std::string("Invalid response: ") + e.what());
    }
}

} // namespace

SelectionClient::SelectionClient(const std::string& socket_path) : socket_path_(socket_path) {}

bool SelectionClient::is_gui_running() const {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    const bool connected = (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    close(fd);
    return connected;
}

std::expected<std::string, std::string> SelectionClient::send_command(const std::string& json_command) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return std::unexpected("Failed to create socket");

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return std::unexpected("GUI not running");
    }

    if (write(fd, json_command.c_str(), json_command.size()) < 0) {
        close(fd);
        return std::unexpected("Failed to send command");
    }

    char buffer[RECV_BUFFER_SIZE];
    const ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    close(fd);

    if (bytes_read <= 0)
        return std::unexpected("No response from GUI");

    buffer[bytes_read] = '\0';
    return std::string(buffer);
}

std::expected<void, std::string> SelectionClient::select_rect(float x0, float y0, float x1, float y1,
                                                               const std::string& mode, int camera_index) {
    const json command = {{"command", "select_rect"}, {"x0", x0},           {"y0", y0},
                          {"x1", x1},                 {"y1", y1},           {"mode", mode},
                          {"camera_index", camera_index}};

    const auto result = send_command(command.dump());
    if (!result)
        return std::unexpected(result.error());

    return parse_response(*result);
}

std::expected<void, std::string> SelectionClient::apply_mask(const std::vector<uint8_t>& mask) {
    const json command = {{"command", "apply_mask"}, {"mask", mask}};

    const auto result = send_command(command.dump());
    if (!result)
        return std::unexpected(result.error());

    return parse_response(*result);
}

std::expected<void, std::string> SelectionClient::deselect_all() {
    const json command = {{"command", "deselect_all"}};

    const auto result = send_command(command.dump());
    if (!result)
        return std::unexpected(result.error());

    return parse_response(*result);
}

} // namespace lfs::mcp
