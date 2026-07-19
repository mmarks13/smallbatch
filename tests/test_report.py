import json

from conftest import make_spec
from smallbatch.report import (
    build_report,
    evidence_summary,
    observed_dominance,
    pairwise_comparisons,
    render_markdown,
    write_report,
)


def candidate(predictions, latency=1, memory=100, size=10):
    return {
        "backend": "tfidf",
        "status": "completed",
        "predictions": predictions,
        "metrics": {"decision_agreement": 1.0},
        "profile": {
            "batch_one_latency_ms": {"p50": latency, "p95": latency},
            "peak_rss_bytes": memory,
            "candidate_owned_bytes": size,
            "required_shared_bytes": 0,
        },
    }


def test_observed_dominance_is_row_level_and_operational():
    spec = make_spec()
    refs = ["urgent", "normal"]
    completed = {
        "a": candidate(refs, 1, 100, 10),
        "b": candidate(["urgent", "urgent"], 2, 200, 20),
    }
    assert observed_dominance(spec, completed, refs) == [{"dominates": "b", "candidate": "a"}]


def test_pairwise_reports_paired_delta_mcnemar_and_no_mae_for_enum():
    spec = make_spec()  # enum urgent/normal
    refs = ["urgent"] * 5
    completed = {"a": candidate(["urgent"] * 5), "b": candidate(["normal"] * 5)}
    pairs = pairwise_comparisons(spec, completed, refs)
    assert len(pairs) == 1
    entry = pairs[0]
    assert (entry["a"], entry["b"]) == ("a", "b")
    assert entry["agreement_delta"] == 1.0
    assert entry["agreement_delta_ci"] == [1.0, 1.0]  # every resample: A right, B wrong
    assert entry["mcnemar_discordant"] == [5, 0]
    assert entry["mcnemar_p"] == 0.0625  # 2 * 0.5**5
    assert "mae_delta" not in entry  # no integer field to score


def test_pairwise_mae_delta_over_complete_case_rows_for_integer_spec():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    refs = [0, 0, 0, 0]
    completed = {"a": candidate([0, 0, 0, 0]), "b": candidate([2, 2, 2, 2])}
    entry = pairwise_comparisons(spec, completed, refs)[0]
    assert entry["mae_delta"] == -2.0  # A minus B
    assert entry["mae_delta_paired_n"] == 4
    assert entry["mae_delta_ci"] == [-2.0, -2.0]
    assert entry["agreement_delta"] == 1.0  # A within-one everywhere, B off by two


def test_markdown_calls_out_robust_pairwise_wins():
    spec = make_spec()
    rows = [{"input": {"title": "t", "body": "b"}, "output": "urgent"} for _ in range(5)]
    completed = {"a": candidate(["urgent"] * 5), "b": candidate(["normal"] * 5)}
    report, _ = build_report(spec, rows, completed, {})
    markdown = render_markdown(report)
    assert "## Paired comparisons" in markdown
    assert "`a` beats `b` on decision agreement" in markdown


def test_public_report_redacts_inputs_and_local_details_keep_them(tmp_path):
    spec = make_spec()
    rows = [
        {"input": {"title": "SECRET", "body": "PRIVATE"}, "output": "urgent"},
        {"input": {"title": "other", "body": "routine"}, "output": "normal"},
    ]
    candidates = {"tfidf": candidate(["normal", "normal"])}
    report, details = build_report(spec, rows, candidates, {})
    path = write_report(tmp_path, report, details)
    assert "SECRET" not in path.read_text()
    assert "SECRET" in (tmp_path / "report_details.local.json").read_text()
    assert report["claims"]["decision_correctness_validated"] is False
    assert report["claims"]["energy_measured"] is False
    assert json.loads(path.read_text())["selection_bias_note"]


def test_markdown_shows_head_training_and_text_sections(tmp_path):
    from smallbatch.report import render_markdown

    diagnostics = {
        "rows": 40,
        "hidden": 64,
        "mean_confidence": 0.72,
        "levels": [
            {"level": 0, "mean_probability": 0.2, "observed_rate": 0.25},
            {"level": 1, "mean_probability": 0.8, "observed_rate": 0.75},
        ],
    }
    tfidf = candidate([1])
    tfidf["head_diagnostics"] = {"score": diagnostics}
    setfit = candidate([1])
    setfit["backend"] = "setfit"
    setfit["field_training"] = {
        "score": {
            "head_diagnostics": None,
            "frozen_vs_tuned": {
                "frozen_dev_within_one": 0.7,
                "best_dev_within_one": 0.85,
                "delta": 0.15,
                "best_epoch": 1,
            },
        }
    }
    lora = candidate([1])
    lora["backend"] = "lora"
    lora["loss_weights"] = {"score": 1.0}
    lora["text_fidelity"] = {
        "bits_per_byte": 1.21,
        "median_example_bpb": 1.1,
        "p90_example_bpb": 2.0,
        "token_nll": 0.9,
        "perplexity": 2.46,
        "top1_accuracy": 0.61,
    }
    lora["metrics"] = dict(lora["metrics"], structural_failures={"char_limit": 2}, n=10)
    lora["training"] = {
        "untuned_baselines": {"score": 2.1, "__format__": 0.4},
        "format_loss_weight": 0.1,
        "best_epoch": 3,
        "epochs_run": 5,
        "best_checkpoint_score": 0.44,
        "stopped_reason": "early_stop(patience=2)",
        "curve": [
            {
                "epoch": 1,
                "checkpoint_score": 0.9,
                "normalized_dev_losses": {"score": 0.9},
            },
            {
                "epoch": 3,
                "checkpoint_score": 0.44,
                "normalized_dev_losses": {"score": 0.44},
            },
        ],
    }

    spec = make_spec(output={"type": "int", "range": [0, 4]})
    report, _ = build_report(
        spec,
        [{"input": {"title": "t", "body": "b"}, "output": 1}],
        {"tfidf": tfidf, "setfit": setfit, "lora": lora},
        {},
    )
    markdown = render_markdown(report)

    # head diagnostics: argmax read, no decoder-selection tables remain
    assert "Ordinal head `score` on dev: 64-hidden capacity, argmax read" in markdown
    assert "| 0 | 0.2000 | 0.2500 |" in markdown
    assert "Decode " not in markdown
    # training configuration and checkpoint evidence are their own sections
    assert "loss weights `score`=1.0" in markdown
    assert "format loss fixed at 0.1" in markdown
    assert "Untuned-base dev losses" in markdown
    assert "Checkpoint: epoch 3 of 5" in markdown
    # text evidence: fidelity framed as reference prediction, never correctness
    assert "held-out teacher-text prediction" in markdown
    assert "not correctness" in markdown
    assert "| 1.2100 |" in markdown
    assert "char_limit=2" in markdown
    assert (
        "Embedding `score`: frozen 0.7000 -> best 0.8500 dev within-one "
        "(delta 0.1500, kept epoch 1)" in markdown
    )


def test_integer_evidence_summary_includes_quality_and_operating_metrics():
    record = candidate([1, 2])
    record["metrics"] = {
        "exact": 0.4,
        "within_one": 0.8,
        "mae": 0.9,
        "p90_absolute_error": 2,
        "max_absolute_error": 3,
        "mean_signed_error": -0.2,
        "pearson_r": 0.7,
        "spearman_rho": 0.6,
        "invalid_rate": 0.0,
    }
    summary = evidence_summary(record)
    for expected in (
        "exact=0.4000",
        "within_one=0.8000",
        "mae=0.9000",
        "p90_error=2",
        "max_error=3",
        "signed_error=-0.2000",
        "pearson=0.7000",
        "spearman=0.6000",
        "p50_ms=1",
        "peak_rss=0.0MiB",
    ):
        assert expected in summary


def test_reference_stability_reports_ceiling_and_relative_agreement():
    spec = make_spec()
    rows = [{"input": {"title": "t", "body": "b"}, "output": "urgent"} for _ in range(5)]
    meta = {
        "teacher_noise": {
            "passes": 2,
            "measured_rows": 40,
            "self_agreement": {"overall": 0.9, "train": 0.9, "dev": None, "eval": 0.9},
            "agreement_counts": {"unanimous": 36, "majority": 4},
            "unresolved": 2,
            "user_resolved": 1,
        }
    }
    completed = {"a": candidate(["urgent"] * 5)}
    report, _ = build_report(spec, rows, completed, {}, data_meta=meta)
    stability = report["reference_stability"]
    assert stability["self_agreement"]["eval"] == 0.9
    assert "ceiling" in stability["note"]
    markdown = render_markdown(report)
    assert "## Reference stability" in markdown
    assert "eval 90.0%" in markdown
    assert "111% of the ceiling" in markdown  # 1.0 agreement vs 0.9 ceiling
    assert "2 items had no stable teacher answer" in markdown
    assert "1 decisions were resolved by the user" in markdown


def test_single_pass_report_states_unknown_ceiling():
    spec = make_spec()
    rows = [{"input": {"title": "t", "body": "b"}, "output": "urgent"}]
    meta = {
        "teacher_noise": {
            "passes": 1,
            "measured_rows": 0,
            "self_agreement": {"overall": None, "train": None, "dev": None, "eval": None},
            "agreement_counts": {},
            "unresolved": 0,
            "user_resolved": 0,
        }
    }
    report, _ = build_report(spec, rows, {"a": candidate(["urgent"])}, {}, data_meta=meta)
    assert "ceiling is unknown" in report["reference_stability"]["note"]
    assert "teacher.passes: 2" in report["reference_stability"]["note"]
    assert "single teacher draw" in render_markdown(report)


def test_imported_decisions_have_no_reference_stability_block():
    spec = make_spec()
    rows = [{"input": {"title": "t", "body": "b"}, "output": "urgent"}]
    report, _ = build_report(
        spec, rows, {"a": candidate(["urgent"])}, {}, data_meta={"teacher_noise": None}
    )
    assert report["reference_stability"] is None
    assert "Reference stability" not in render_markdown(report)
