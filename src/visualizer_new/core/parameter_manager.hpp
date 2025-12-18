/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/parameters.hpp"
#include <expected>
#include <string>
#include <string_view>

namespace lfs::vis {

// Session defaults set once at startup (CLI > --config > JSON), current params are user-editable.
class ParameterManager {
public:
    std::expected<void, std::string> ensureLoaded();

    [[nodiscard]] lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy);
    [[nodiscard]] const lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy) const;

    [[nodiscard]] lfs::core::param::LoadingParams& getLoadingParams() { return loading_params_; }
    [[nodiscard]] const lfs::core::param::LoadingParams& getLoadingParams() const { return loading_params_; }

    // Reset current to session defaults
    void resetToDefaults(std::string_view strategy = "");

    // Set session defaults (called once at startup)
    void setSessionDefaults(const lfs::core::param::OptimizationParameters& params);

    [[nodiscard]] const std::string& getActiveStrategy() const { return active_strategy_; }
    void setActiveStrategy(std::string_view strategy);

    [[nodiscard]] lfs::core::param::OptimizationParameters& getActiveParams();
    [[nodiscard]] const lfs::core::param::OptimizationParameters& getActiveParams() const;

    [[nodiscard]] lfs::core::param::TrainingParameters createForDataset(
        const std::filesystem::path& data_path,
        const std::filesystem::path& output_path) const;

    [[nodiscard]] bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;
    bool session_defaults_set_ = false;
    std::string active_strategy_ = "mcmc";

    // Session defaults (immutable after startup)
    lfs::core::param::OptimizationParameters mcmc_session_;
    lfs::core::param::OptimizationParameters default_session_;

    // Current params (user-editable)
    lfs::core::param::OptimizationParameters mcmc_current_;
    lfs::core::param::OptimizationParameters default_current_;

    lfs::core::param::LoadingParams loading_params_;
};

} // namespace lfs::vis
