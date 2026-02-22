#!/usr/bin/env python3
"""
Add Prometheus /metrics endpoint to PowerInfer examples/server/server.cpp.

Ported from smallthinker/tools/server/server.cpp which already has full
metrics support. The legacy examples/server uses a simpler synchronous
queue, so the handler reads state directly under mutex rather than posting
a task.

Adds:
  - server_metrics struct (counters + bucket tracking)
  - metrics / endpoint_metrics fields on llama_server_context
  - metrics.init() call in main()
  - metrics.on_decoded() after each llama_decode()
  - metrics.on_prompt_eval() after prompt processing
  - metrics.on_prediction() before send_final_response()
  - --metrics CLI flag
  - GET /metrics HTTP handler (Prometheus text format 0.0.4)
"""

import os
import re
import sys
import shutil


def apply_fix(filepath, pattern, replacement, description, flags=re.DOTALL, backup_suffix='.backup.metrics'):
    if not os.path.exists(filepath):
        print(f"  ERROR: file not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if not re.search(pattern, content, flags):
        print(f"  SKIP (pattern not found): {description}")
        return False

    new_content = re.sub(pattern, replacement, content, count=1, flags=flags)
    if new_content == content:
        print(f"  SKIP (no change): {description}")
        return False

    if not os.path.exists(filepath + backup_suffix):
        shutil.copy2(filepath, filepath + backup_suffix)

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"  OK: {description}")
    return True


def main():
    powerinfer_dir = sys.argv[1] if len(sys.argv) > 1 else '/opt/powerinfer'
    server_cpp = os.path.join(powerinfer_dir, 'examples', 'server', 'server.cpp')

    print("=== Applying Prometheus metrics support to PowerInfer server ===")
    print(f"Target: {server_cpp}")

    applied = 0

    # ------------------------------------------------------------------
    # 1. Insert server_metrics struct before llama_server_context
    # ------------------------------------------------------------------
    server_metrics_struct = r'''struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;
    uint64_t n_tokens_predicted        = 0;
    uint64_t t_tokens_generation       = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const llama_client_slot & slot) {
        n_prompt_tokens_processed_total += slot.num_prompt_tokens_processed;
        t_prompt_processing_total       += (uint64_t)slot.t_prompt_processing;
        n_prompt_tokens_processed       += slot.num_prompt_tokens_processed;
        t_prompt_processing             += (uint64_t)slot.t_prompt_processing;
    }

    void on_prediction(const llama_client_slot & slot) {
        n_tokens_predicted_total  += slot.n_decoded;
        t_tokens_generation_total += (uint64_t)slot.t_token_generation;
        n_tokens_predicted        += slot.n_decoded;
        t_tokens_generation       += (uint64_t)slot.t_token_generation;
    }

    void on_decoded(const std::vector<llama_client_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

'''

    applied += apply_fix(
        server_cpp,
        r'(struct llama_server_context\s*\n\{)',
        server_metrics_struct + r'\1',
        "Add server_metrics struct",
    )

    # ------------------------------------------------------------------
    # 2. Add metrics fields to llama_server_context
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(    std::mutex mutex_results;)',
        r'\1\n\n    server_metrics metrics;\n    bool endpoint_metrics = false;',
        "Add metrics fields to llama_server_context",
    )

    # ------------------------------------------------------------------
    # 3. Initialize metrics in main() after llama.initialize()
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(    llama\.initialize\(\);)',
        r'\1\n    llama.metrics.init();',
        "Call metrics.init() in main()",
    )

    # ------------------------------------------------------------------
    # 4. on_decoded() after each successful llama_decode()
    #    Matches the retry-shrink block that ends with continue;
    #    followed by the per-slot loop.
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(                i -= n_batch;\s*\n\s*continue;\s*\n\s*\}\s*\n\n)(            for \(auto & slot : slots\))',
        r'\1            metrics.on_decoded(slots);\n\n\2',
        "Call metrics.on_decoded() after llama_decode()",
    )

    # ------------------------------------------------------------------
    # 5. on_prompt_eval() after t_prompt_processing is set
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(                    slot\.t_prompt_processing = \(slot\.t_start_genereration - slot\.t_start_process_prompt\) / 1e3;\s*\n)',
        r'\1                    metrics.on_prompt_eval(slot);\n',
        "Call metrics.on_prompt_eval() after prompt timing",
    )

    # ------------------------------------------------------------------
    # 6. on_prediction() after slot.release(), before send_final_response
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(                    slot\.release\(\);\s*\n)(                    slot\.print_timings\(\);\s*\n\s*send_final_response\(slot\);)',
        r'\1                    metrics.on_prediction(slot);\n\2',
        "Call metrics.on_prediction() before send_final_response()",
    )

    # ------------------------------------------------------------------
    # 7. --metrics CLI flag (insert before the catch-all else)
    # ------------------------------------------------------------------
    applied += apply_fix(
        server_cpp,
        r'(        else\s*\n        \{\s*\n            fprintf\(stderr, "error: unknown argument)',
        r'        else if (arg == "--metrics")\n        {\n            llama.endpoint_metrics = true;\n        }\n\1',
        "Add --metrics CLI flag",
    )

    # ------------------------------------------------------------------
    # 8. GET /metrics HTTP handler (insert before svr.set_logger)
    # ------------------------------------------------------------------
    metrics_handler = r'''
    svr.Get("/metrics", [&llama](const httplib::Request &, httplib::Response & res)
            {
                if (!llama.endpoint_metrics)
                {
                    res.status = 404;
                    res.set_content("{\"error\":\"metrics endpoint disabled, start with --metrics\"}", "application/json");
                    return;
                }

                std::lock_guard<std::mutex> lock_tasks(llama.mutex_tasks);

                int n_idle_slots       = 0;
                int n_processing_slots = 0;
                for (const auto & slot : llama.slots)
                {
                    if (slot.is_processing()) { n_processing_slots++; }
                    else                      { n_idle_slots++;       }
                }

                const server_metrics & m = llama.metrics;

                std::string out;
                const auto add_metric = [&](const char * name, const char * help, const char * type, double value)
                {
                    out += std::string("# HELP llamacpp:") + name + " " + help + "\n";
                    out += std::string("# TYPE llamacpp:") + name + " " + type + "\n";
                    out += std::string("llamacpp:") + name + " " + std::to_string(value) + "\n";
                };

                add_metric("prompt_tokens_total",
                    "Number of prompt tokens processed.", "counter",
                    (double)m.n_prompt_tokens_processed_total);
                add_metric("prompt_seconds_total",
                    "Prompt processing time.", "counter",
                    m.t_prompt_processing_total / 1.e3);
                add_metric("tokens_predicted_total",
                    "Number of generation tokens processed.", "counter",
                    (double)m.n_tokens_predicted_total);
                add_metric("tokens_predicted_seconds_total",
                    "Generation processing time.", "counter",
                    m.t_tokens_generation_total / 1.e3);
                add_metric("n_decode_total",
                    "Total number of llama_decode() calls.", "counter",
                    (double)m.n_decode_total);
                add_metric("prompt_tokens_seconds",
                    "Average prompt throughput in tokens/s.", "gauge",
                    m.t_prompt_processing > 0
                        ? 1.e3 / m.t_prompt_processing * m.n_prompt_tokens_processed
                        : 0.0);
                add_metric("predicted_tokens_seconds",
                    "Average generation throughput in tokens/s.", "gauge",
                    m.t_tokens_generation > 0
                        ? 1.e3 / m.t_tokens_generation * m.n_tokens_predicted
                        : 0.0);
                add_metric("requests_processing",
                    "Number of requests currently being processed.", "gauge",
                    (double)n_processing_slots);
                add_metric("requests_deferred",
                    "Number of requests deferred.", "gauge",
                    (double)llama.queue_tasks.size());
                add_metric("n_busy_slots_per_decode",
                    "Average number of slots processing per llama_decode() call.", "gauge",
                    m.n_decode_total > 0
                        ? (double)m.n_busy_slots_total / m.n_decode_total
                        : 0.0);

                res.set_header("Process-Start-Time-Unix", std::to_string(m.t_start / 1000000));
                res.set_content(out, "text/plain; version=0.0.4");
            });

'''

    applied += apply_fix(
        server_cpp,
        r'(    svr\.set_logger\(log_server_request\);)',
        metrics_handler + r'\1',
        "Add GET /metrics HTTP handler",
    )

    print(f"\n=== Done: {applied}/8 fixes applied ===")
    if applied < 8:
        print("WARNING: some fixes were skipped — check patterns against your PowerInfer version")
    return 0 if applied > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
