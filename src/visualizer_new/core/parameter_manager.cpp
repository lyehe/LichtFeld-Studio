/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "parameter_manager.hpp"
#include "core_new/logger.hpp"

namespace lfs::vis {

namespace {
    constexpr const char* MCMC_CONFIG_FILE = "mcmc_optimization_params.json";
    constexpr const char* DEFAULT_CONFIG_FILE = "default_optimization_params.json";
    constexpr const char* LOADING_CONFIG_FILE = "loading_params.json";
} // namespace

std::expected<void, std::string> ParameterManager::ensureLoaded() {
    if (loaded_) return {};

    const auto mcmc_path = lfs::core::param::get_parameter_file_path(MCMC_CONFIG_FILE);
    auto mcmc_result = lfs::core::param::read_optim_params_from_json(mcmc_path);
    if (!mcmc_result) {
        return std::unexpected("Failed to load MCMC params: " + mcmc_result.error());
    }
    mcmc_session_ = std::move(*mcmc_result);
    mcmc_current_ = mcmc_session_;

    const auto default_path = lfs::core::param::get_parameter_file_path(DEFAULT_CONFIG_FILE);
    auto default_result = lfs::core::param::read_optim_params_from_json(default_path);
    if (!default_result) {
        return std::unexpected("Failed to load default params: " + default_result.error());
    }
    default_session_ = std::move(*default_result);
    default_current_ = default_session_;

    const auto loading_path = lfs::core::param::get_parameter_file_path(LOADING_CONFIG_FILE);
    auto loading_result = lfs::core::param::read_loading_params_from_json(loading_path);
    if (!loading_result) {
        return std::unexpected("Failed to load loading params: " + loading_result.error());
    }
    loading_params_ = std::move(*loading_result);

    loaded_ = true;
    return {};
}

lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) {
    return (strategy == "mcmc") ? mcmc_current_ : default_current_;
}

const lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) const {
    return (strategy == "mcmc") ? mcmc_current_ : default_current_;
}

void ParameterManager::resetToDefaults(const std::string_view strategy) {
    if (strategy.empty() || strategy == "mcmc") {
        mcmc_current_ = mcmc_session_;
    }
    if (strategy.empty() || strategy == "default") {
        default_current_ = default_session_;
    }
}

void ParameterManager::setSessionDefaults(const lfs::core::param::OptimizationParameters& params) {
    if (const auto result = ensureLoaded(); !result) {
        LOG_ERROR("Failed to load params: {}", result.error());
        return;
    }

    if (session_defaults_set_) {
        return;
    }

    if (!params.strategy.empty()) {
        setActiveStrategy(params.strategy);
    }

    if (active_strategy_ == "mcmc") {
        mcmc_session_ = params;
        mcmc_current_ = params;
    } else {
        default_session_ = params;
        default_current_ = params;
    }

    session_defaults_set_ = true;
    LOG_INFO("Session params: strategy={}, iter={}, max_cap={}, sh={}",
             params.strategy, params.iterations, params.max_cap, params.sh_degree);
}

void ParameterManager::setActiveStrategy(const std::string_view strategy) {
    if (strategy == "mcmc" || strategy == "default") {
        active_strategy_ = std::string(strategy);
    }
}

lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() {
    return getCurrentParams(active_strategy_);
}

const lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() const {
    return getCurrentParams(active_strategy_);
}

lfs::core::param::TrainingParameters ParameterManager::createForDataset(
    const std::filesystem::path& data_path,
    const std::filesystem::path& output_path) const {

    lfs::core::param::TrainingParameters params;
    params.optimization = getActiveParams();
    params.dataset.loading_params = loading_params_;
    params.dataset.data_path = data_path;
    params.dataset.output_path = output_path;
    return params;
}

} // namespace lfs::vis
