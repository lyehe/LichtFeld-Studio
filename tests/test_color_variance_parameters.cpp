/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>

namespace {
    namespace param = lfs::core::param;

    TEST(ColorVarianceParameters, AllStrategiesRoundTripAndScaleWarmup) {
        for (const auto* strategy : {"mcmc", "mrnf", "igs+"}) {
            auto p = param::OptimizationParameters::defaults_for_strategy(strategy);
            EXPECT_FALSE(p.use_color_variance);
            EXPECT_EQ(p.color_variance_weight, 1.0f);
            EXPECT_EQ(p.color_variance_weight_at(1), 0.0f);
            p.use_exposure_correction = false;
            p.use_color_variance = true;
            p.color_variance_weight = 1.0f;
            p.color_variance_warmup_iters = 15'000;
            EXPECT_TRUE(p.validate().empty()) << p.validate();
            EXPECT_EQ(p.color_variance_weight_at(15'000), 0.0f);
            EXPECT_EQ(p.color_variance_weight_at(15'001), 1.0f);
            auto restored = param::OptimizationParameters::from_json(p.to_json());
            EXPECT_TRUE(restored.use_color_variance);
            EXPECT_EQ(restored.color_variance_weight, 1.0f);
            EXPECT_EQ(restored.color_variance_warmup_iters, 15'000u);
            restored.scale_steps(2.0f);
            EXPECT_EQ(restored.color_variance_warmup_iters, 30'000u);
            EXPECT_EQ(restored.color_variance_weight_at(30'000), 0.0f);
            EXPECT_EQ(restored.color_variance_weight_at(30'001), 1.0f);
            auto legacy = p.to_json();
            legacy.erase("use_color_variance");
            legacy.erase("color_variance_weight");
            legacy.erase("color_variance_warmup_iters");
            const auto old = param::OptimizationParameters::from_json(legacy);
            EXPECT_FALSE(old.use_color_variance);
            EXPECT_EQ(old.color_variance_weight, 1.0f);
            EXPECT_EQ(old.color_variance_weight_at(1), 0.0f);
            EXPECT_EQ(old.color_variance_warmup_iters, 0u);
        }
    }

    TEST(ColorVarianceParameters, RejectsInvalidWeightsAndUnsupportedModes) {
        auto p = param::OptimizationParameters::mcmc_defaults();
        for (float value : {-1.0f, 9.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            p.color_variance_weight = value;
            EXPECT_FALSE(p.validate().empty());
        }
        p.color_variance_weight = 1.0f;
        p.use_color_variance = true;
        EXPECT_TRUE(p.validate().empty());
        for (bool* flag : {&p.gut, &p.mip_filter, &p.use_ppisp, &p.use_bilateral_grid, &p.use_exposure_correction, &p.enable_sparsity}) {
            *flag = true;
            EXPECT_FALSE(p.validate().empty());
            *flag = false;
        }
        p.mask_mode = param::MaskMode::Ignore;
        EXPECT_FALSE(p.validate().empty());
        p.mask_mode = param::MaskMode::None;
        p.color_variance_warmup_iters = p.iterations;
        EXPECT_FALSE(p.validate().empty());
        p.use_color_variance = false;
        p.mip_filter = true;
        EXPECT_TRUE(p.validate().empty());
    }
    TEST(ColorVarianceParameters, TogglePreservesCustomWeight) {
        auto p = param::OptimizationParameters::mcmc_defaults();
        EXPECT_EQ(p.color_variance_weight_at(1), 0.0f);
        p.use_color_variance = true;
        EXPECT_EQ(p.color_variance_weight_at(1), 1.0f);
        p.color_variance_weight = 2.5f;
        p.use_color_variance = false;
        EXPECT_EQ(p.color_variance_weight_at(1), 0.0f);
        p.use_color_variance = true;
        EXPECT_EQ(p.color_variance_weight_at(1), 2.5f);
    }

    TEST(ColorVarianceParameters, LegacyConfigsPreserveEnabledAndDisabledTraining) {
        auto config = param::OptimizationParameters::mcmc_defaults().to_json();
        config.erase("use_color_variance");
        config["color_variance_weight"] = 2.0f;
        auto restored = param::OptimizationParameters::from_json(config);
        EXPECT_TRUE(restored.use_color_variance);
        EXPECT_EQ(restored.color_variance_weight_at(1), 2.0f);
        config["color_variance_weight"] = 0.0f;
        restored = param::OptimizationParameters::from_json(config);
        EXPECT_FALSE(restored.use_color_variance);
        EXPECT_EQ(restored.color_variance_weight, 1.0f);
        EXPECT_EQ(restored.color_variance_weight_at(1), 0.0f);
        config["use_color_variance"] = false;
        config["color_variance_weight"] = 2.0f;
        restored = param::OptimizationParameters::from_json(config);
        EXPECT_FALSE(restored.use_color_variance);
        EXPECT_EQ(restored.color_variance_weight, 2.0f);
        config["use_color_variance"] = true;
        config.erase("color_variance_weight");
        restored = param::OptimizationParameters::from_json(config);
        EXPECT_EQ(restored.color_variance_weight_at(1), 1.0f);
    }

    TEST(ColorVarianceParameters, ExplicitOverridesRespectEnableFlag) {
        auto params = param::TrainingParameters{};
        param::ExplicitTrainingOverrides overrides;
        overrides.optimization_json = R"({"color_variance_weight":2.0})";
        param::apply_explicit_training_overrides(params, overrides);
        EXPECT_TRUE(params.optimization.use_color_variance);
        EXPECT_EQ(params.optimization.color_variance_weight_at(1), 2.0f);
        overrides.optimization_json = R"({"use_color_variance":false,"color_variance_weight":3.0})";
        param::apply_explicit_training_overrides(params, overrides);
        EXPECT_FALSE(params.optimization.use_color_variance);
        EXPECT_EQ(params.optimization.color_variance_weight, 3.0f);
        overrides.optimization_json = R"({"use_color_variance":true})";
        param::apply_explicit_training_overrides(params, overrides);
        EXPECT_EQ(params.optimization.color_variance_weight_at(1), 3.0f);
    }
} // namespace
