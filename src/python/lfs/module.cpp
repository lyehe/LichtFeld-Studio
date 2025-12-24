#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>

#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/logger.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/istrategy.hpp"
#include "training/trainer.hpp"

#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

    using lfs::training::ArgValue;
    using lfs::training::Command;
    using lfs::training::CommandCenter;
    using lfs::training::CommandTarget;
    using lfs::training::ControlBoundary;
    using lfs::training::ControlHook;
    using lfs::training::HookContext;
    using lfs::training::Selection;
    using lfs::training::SelectionKind;

    // Forward decl
    ArgValue to_numeric_value(const nb::handle& obj);

    struct PyHookContext {
        int iter;
        float loss;
        std::size_t num_splats;
        bool is_refining;
    };

    struct PyAttributeHandle {
        std::string attribute;
        Selection selection;

        PyAttributeHandle slice(int64_t start, int64_t end) const {
            return PyAttributeHandle{attribute, Selection{SelectionKind::Range, start, end, {}}};
        }

        PyAttributeHandle all() const {
            return PyAttributeHandle{attribute, Selection{SelectionKind::All, 0, 0, {}}};
        }

        PyAttributeHandle indices(const std::vector<int64_t>& idx) const {
            return PyAttributeHandle{attribute, Selection{SelectionKind::Indices, 0, 0, idx}};
        }

        PyAttributeHandle refine(nb::handle key) const {
            if (nb::isinstance<nb::slice>(key)) {
                nb::slice s = nb::cast<nb::slice>(key);
                const size_t dummy_len = std::numeric_limits<size_t>::max();

                auto [start, stop, step, slicelen] = s.compute(dummy_len);
                
                if (step != 1) {
                    throw nb::value_error("Only step=1 slices are supported");
                }

                int64_t s64 = static_cast<int64_t>(start);
                int64_t e64 = (stop == dummy_len)
                                ? -1
                                : static_cast<int64_t>(stop);

                return slice(s64, e64);
            }
            if (nb::isinstance<nb::int_>(key)) {
                auto idx = nb::cast<int64_t>(key);
                if (idx < 0) throw nb::value_error("Negative indices are not supported");
                return slice(idx, idx + 1);
            }
            if (nb::isinstance<nb::list>(key) || nb::isinstance<nb::tuple>(key)) {
                std::vector<int64_t> idx;
                nb::iterable it = nb::cast<nb::iterable>(key);
                for (auto item : it) {
                    idx.push_back(nb::cast<int64_t>(item));
                }
                return indices(idx);
            }
            throw nb::type_error("Unsupported index type; use slice, int, or sequence of ints");
        }

        void set(nb::handle value) const {
            Command cmd{
                .target = CommandTarget::Model,
                .op = "set_attribute",
                .selection = selection,
                .args = {{"attribute", ArgValue{attribute}}, {"value", to_numeric_value(value)}}};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }

        void scale(double factor) const {
            Command cmd{
                .target = CommandTarget::Model,
                .op = "scale_attribute",
                .selection = selection,
                .args = {{"attribute", ArgValue{attribute}}, {"factor", ArgValue{factor}}}};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }

        void clamp(std::optional<double> min, std::optional<double> max) const {
            Command cmd{
                .target = CommandTarget::Model,
                .op = "clamp_attribute",
                .selection = selection,
                .args = {{"attribute", ArgValue{attribute}}}};
            if (min) cmd.args.emplace("min", ArgValue{*min});
            if (max) cmd.args.emplace("max", ArgValue{*max});
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }
    };

    CommandTarget parse_target(const std::string& t) {
        if (t == "model") return CommandTarget::Model;
        if (t == "optimizer") return CommandTarget::Optimizer;
        return CommandTarget::Session;
    }

    struct PySelection {
        Selection sel;
    };

    ArgValue to_argvalue(const nb::handle& obj) {
        if (nb::isinstance<nb::bool_>(obj)) {
            return ArgValue{static_cast<bool>(nb::cast<bool>(obj))};
        }
        if (nb::isinstance<nb::int_>(obj)) {
            return ArgValue{static_cast<int64_t>(nb::cast<int64_t>(obj))};
        }
        if (nb::isinstance<nb::float_>(obj)) {
            return ArgValue{static_cast<double>(nb::cast<double>(obj))};
        }
        if (nb::isinstance<nb::str>(obj)) {
            return ArgValue{nb::cast<std::string>(obj)};
        }
        if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
            std::vector<double> vals;
            nb::iterable it = nb::cast<nb::iterable>(obj);
            for (auto item : it) {
                vals.push_back(nb::cast<double>(item));
            }
            return ArgValue{std::move(vals)};
        }
        throw nb::type_error("Unsupported arg type");
    }

    // For model attribute values we only accept scalar/iterable numeric → double/vector<double>
    ArgValue to_numeric_value(const nb::handle& obj) {
        if (nb::isinstance<nb::float_>(obj) || nb::isinstance<nb::int_>(obj)) {
            return ArgValue{static_cast<double>(nb::cast<double>(obj))};
        }
        if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
            std::vector<double> vals;
            nb::iterable it = nb::cast<nb::iterable>(obj);
            for (auto item : it) {
                vals.push_back(nb::cast<double>(item));
            }
            return ArgValue{std::move(vals)};
        }
        throw nb::type_error("Value must be int, float, or sequence of numbers");
    }

    std::unordered_map<std::string, ArgValue> dict_to_args(const nb::dict& d) {
        std::unordered_map<std::string, ArgValue> out;
        for (auto [k, v] : d) {
            out.emplace(nb::cast<std::string>(k), to_argvalue(v));
        }
        return out;
    }

    struct PyCommandBuilder {
        void enqueue(const std::string& target,
                     const std::string& op,
                     std::optional<PySelection> sel,
                     nb::dict args) {
            Command cmd{
                .target = parse_target(target),
                .op = op,
                .selection = sel ? sel->sel : Selection{},
                .args = dict_to_args(args)};

            // Defer execution to the training thread for CUDA safety
            CommandCenter::instance().enqueue_command(cmd);
        }

        void flush() {}
    };

    std::size_t register_hook(ControlHook hook, nb::callable cb) {
        if (!cb) return 0;
        nb::object ocb = nb::cast<nb::object>(cb);
        LOG_INFO("Python hook registered for hook {}", static_cast<int>(hook));
        return ControlBoundary::instance().register_callback(hook, [ocb, hook](const HookContext& ctx) {
            nb::gil_scoped_acquire guard;
            LOG_DEBUG("Python hook invoke hook={} iter={} loss={} gauss={} refining={}",
                      static_cast<int>(hook), ctx.iteration, ctx.loss, ctx.num_gaussians, ctx.is_refining);
            try {
                nb::dict d;
                d["iter"] = ctx.iteration;
                d["loss"] = ctx.loss;
                d["num_splats"] = ctx.num_gaussians;
                d["is_refining"] = ctx.is_refining;
                ocb(d);
            } catch (const std::exception& e) {
                LOG_ERROR("Python hook threw std::exception: {}", e.what());
            } catch (...) {
                LOG_ERROR("Python hook threw unknown exception");
            }
        });
    }

    struct PyModelView {
        PySelection auto_select(const std::optional<PySelection>& sel) const {
            return sel ? *sel : select_all();
        }

        PySelection select_all() const { return PySelection{Selection{SelectionKind::All, 0, 0, {}}}; }
        PySelection select_range(int64_t start, int64_t end) const { return PySelection{Selection{SelectionKind::Range, start, end, {}}}; }
        PySelection select_indices(const std::vector<int64_t>& idx) const { return PySelection{Selection{SelectionKind::Indices, 0, 0, idx}}; }

        PyAttributeHandle get_attr(const std::string& name) const {
            return PyAttributeHandle{name, Selection{SelectionKind::All, 0, 0, {}}};
        }

        nb::list attributes() const {
            nb::list out;
            for (const auto& f : CommandCenter::instance().mutables(CommandTarget::Model)) {
                nb::dict d;
                d["name"] = f.name;
                d["shape"] = f.shape;
                d["writable"] = f.writable;
                d["description"] = f.description;
                out.append(std::move(d));
            }
            return out;
        }

        void set(const std::string& attribute, nb::handle value, std::optional<PySelection> sel = std::nullopt) const {
            PyAttributeHandle{attribute, auto_select(sel).sel}.set(value);
        }

        void scale(const std::string& attribute, double factor, std::optional<PySelection> sel = std::nullopt) const {
            PyAttributeHandle{attribute, auto_select(sel).sel}.scale(factor);
        }

        void clamp(const std::string& attribute, std::optional<double> min = std::nullopt, std::optional<double> max = std::nullopt, std::optional<PySelection> sel = std::nullopt) const {
            PyAttributeHandle{attribute, auto_select(sel).sel}.clamp(min, max);
        }
    };

    struct PyOptimizerView {
        nb::list params() const {
            nb::list out;
            auto snap = CommandCenter::instance().snapshot();
            if (!snap.trainer) {
                return out;
            }
            auto& opt = snap.trainer->get_strategy_mutable().get_optimizer();
            nb::dict g;
            g["name"] = "default";
            g["lr"] = opt.get_lr();
            out.append(std::move(g));
            return out;
        }

        void set_lr(double value) const {
            Command cmd{.target = CommandTarget::Optimizer, .op = "set_lr", .selection = Selection{SelectionKind::All, 0, 0, {}}, .args = { {"value", ArgValue{value}} }};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }

        void scale_lr(double factor) const {
            Command cmd{.target = CommandTarget::Optimizer, .op = "scale_lr", .selection = Selection{SelectionKind::All, 0, 0, {}}, .args = { {"factor", ArgValue{factor}} }};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }
    };

    struct PyIntrospectionView {
        nb::list operations(std::optional<std::string> target) const {
            std::optional<CommandTarget> ct;
            if (target) ct = parse_target(*target);
            nb::list out;
            for (const auto& op : CommandCenter::instance().operations(ct)) {
                nb::dict d;
                d["name"] = op.name;
                d["target"] = (op.target == CommandTarget::Model) ? "model" : (op.target == CommandTarget::Optimizer ? "optimizer" : "session");
                nb::list args;
                for (const auto& a : op.args) {
                    nb::dict ad;
                    ad["name"] = a.name;
                    ad["required"] = a.required;
                    if (a.description) ad["description"] = *a.description;
                    args.append(std::move(ad));
                }
                d["args"] = std::move(args);
                out.append(std::move(d));
            }
            return out;
        }

        nb::list mutables(std::optional<std::string> target) const {
            std::optional<CommandTarget> ct;
            if (target) ct = parse_target(*target);
            nb::list out;
            for (const auto& f : CommandCenter::instance().mutables(ct)) {
                nb::dict d;
                d["name"] = f.name;
                d["shape"] = f.shape;
                d["writable"] = f.writable;
                d["description"] = f.description;
                out.append(std::move(d));
            }
            return out;
        }
    };

    struct PySessionView {
        int iteration() const { return CommandCenter::instance().snapshot().iteration; }
        int max_iterations() const { return CommandCenter::instance().snapshot().max_iterations; }
        float loss() const { return CommandCenter::instance().snapshot().loss; }
        bool running() const { return CommandCenter::instance().snapshot().is_running; }
        bool paused() const { return CommandCenter::instance().snapshot().is_paused; }
        bool stopping() const { return CommandCenter::instance().snapshot().stop_requested; }
        std::size_t num_splats() const { return CommandCenter::instance().snapshot().num_gaussians; }

        PyModelView model() const { return PyModelView{}; }
        PyOptimizerView optimizer() const { return PyOptimizerView{}; }
        PyCommandBuilder commands() const { return PyCommandBuilder{}; }
        PyIntrospectionView introspect() const { return PyIntrospectionView{}; }

        void pause() const {
            Command cmd{.target = CommandTarget::Session, .op = "pause", .selection = Selection{}, .args = {}};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }
        void resume() const {
            Command cmd{.target = CommandTarget::Session, .op = "resume", .selection = Selection{}, .args = {}};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }
        void request_stop() const {
            Command cmd{.target = CommandTarget::Session, .op = "request_stop", .selection = Selection{}, .args = {}};
            auto res = CommandCenter::instance().execute(cmd);
            if (!res) throw nb::value_error(res.error().c_str());
        }
    };

} // namespace

NB_MODULE(lichtfeld, m) {
    m.doc() = "LichtFeld embedded Python control module (command-based)";

    nb::class_<PySelection>(m, "Selection");

    nb::class_<PyAttributeHandle>(m, "Attribute")
        .def("__getitem__", &PyAttributeHandle::refine, nb::arg("key"))
        .def("all", &PyAttributeHandle::all, "Select all elements")
        .def("indices", &PyAttributeHandle::indices, nb::arg("indices"), "Select specific indices")
        .def("set", &PyAttributeHandle::set, nb::arg("value"), "Set attribute values for this selection")
        .def("scale", &PyAttributeHandle::scale, nb::arg("factor"), "Scale attribute values for this selection")
        .def("clamp", &PyAttributeHandle::clamp, nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), "Clamp attribute values for this selection");

    nb::class_<PyModelView>(m, "Model")
        .def("select_all", &PyModelView::select_all, "Select all splats")
        .def("select_range", &PyModelView::select_range, nb::arg("start"), nb::arg("end"), "Select a range [start, end) of splats")
        .def("select_indices", &PyModelView::select_indices, nb::arg("indices"), "Select specific splat indices")
        .def("__getattr__", &PyModelView::get_attr, nb::arg("name"), "Get an attribute handle (supports slicing)")
        .def("attributes", &PyModelView::attributes, "List mutable attributes")
        .def("set", &PyModelView::set, nb::arg("attribute"), nb::arg("value"), nb::arg("where") = nb::none(), "Set attribute (scalar or vector) for selection")
        .def("scale", &PyModelView::scale, nb::arg("attribute"), nb::arg("factor"), nb::arg("where") = nb::none(), "Scale attribute for selection")
        .def("clamp", &PyModelView::clamp, nb::arg("attribute"), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("where") = nb::none(), "Clamp attribute for selection");

    nb::class_<PyOptimizerView>(m, "Optimizer")
        .def("params", &PyOptimizerView::params, "List optimizer parameter groups")
        .def("set_lr", &PyOptimizerView::set_lr, nb::arg("value"), "Set global learning rate")
        .def("scale_lr", &PyOptimizerView::scale_lr, nb::arg("factor"), "Scale global learning rate");

    nb::class_<PyCommandBuilder>(m, "Commands")
        .def("enqueue", &PyCommandBuilder::enqueue, nb::arg("target"), nb::arg("op"), nb::arg("selector") = nb::none(), nb::arg("args") = nb::dict(), "Submit an immediate command")
        .def("flush", &PyCommandBuilder::flush, "No-op flush for batching compatibility");

    nb::class_<PyIntrospectionView>(m, "Introspect")
        .def("operations", &PyIntrospectionView::operations, nb::arg("target") = nb::none(), "Supported operations")
        .def("mutables", &PyIntrospectionView::mutables, nb::arg("target") = nb::none(), "Mutable fields");

    nb::class_<PySessionView>(m, "Session")
        .def(nb::init<>())
        .def_prop_ro("iter", &PySessionView::iteration)
        .def_prop_ro("max_iters", &PySessionView::max_iterations)
        .def_prop_ro("loss", &PySessionView::loss)
        .def_prop_ro("num_splats", &PySessionView::num_splats)
        .def_prop_ro("running", &PySessionView::running)
        .def_prop_ro("paused", &PySessionView::paused)
        .def_prop_ro("stopping", &PySessionView::stopping)
        .def_prop_ro("model", &PySessionView::model)
        .def_prop_ro("optimizer", &PySessionView::optimizer)
        .def_prop_ro("commands", &PySessionView::commands)
        .def_prop_ro("introspect", &PySessionView::introspect)
        .def("pause", &PySessionView::pause)
        .def("resume", &PySessionView::resume)
        .def("request_stop", &PySessionView::request_stop);

    // Hook registration
    nb::enum_<ControlHook>(m, "ControlHook")
        .value("TrainingStart", ControlHook::TrainingStart)
        .value("IterationStart", ControlHook::IterationStart)
        .value("PreOptimizerStep", ControlHook::PreOptimizerStep)
        .value("PostStep", ControlHook::PostStep)
        .value("TrainingEnd", ControlHook::TrainingEnd);

    m.def("on_training_start", [](nb::callable cb) {
        return register_hook(ControlHook::TrainingStart, cb);
    }, "Register a callback invoked at the beginning of training");

    m.def("on_iteration_start", [](nb::callable cb) {
        return register_hook(ControlHook::IterationStart, cb);
    }, "Register a callback invoked at the start of each iteration");

    m.def("on_pre_optimizer_step", [](nb::callable cb) {
        return register_hook(ControlHook::PreOptimizerStep, cb);
    }, "Register a callback invoked before each optimizer step");

    m.def("on_post_step", [](nb::callable cb) {
        return register_hook(ControlHook::PostStep, cb);
    }, "Register a callback invoked after each optimizer step");

    m.def("on_training_end", [](nb::callable cb) {
        return register_hook(ControlHook::TrainingEnd, cb);
    }, "Register a callback invoked at the end of training");

    // Convenience factory
    m.def("session", []() { return PySessionView{}; }, "Get the active training session view");
}
