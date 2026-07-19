"""Shared task data and builders for the GPU tier.

Not a conftest: tests import this by name so it can never collide with the
main suite's conftest module when both run in one pytest invocation.
"""

from __future__ import annotations

import os

BASE_MODEL = os.environ.get("SMALLBATCH_GPU_BASE", "ibm-granite/granite-4.0-350m")


# --- shared task data: the triage recipe the v0.3 smoke test validated ------

CASES = [
    (4, "site down", "every request returns 503 for all customers", "Full outage: all requests fail."),
    (4, "data loss", "nightly job deleted customer records", "Customer records were destroyed."),
    (4, "outage", "checkout broken worldwide since 09:00", "Checkout is failing for everyone."),
    (4, "database down", "primary db unreachable, writes failing", "Primary database is unreachable."),
    (4, "payments failing", "all card payments rejected", "No payment can complete."),
    (4, "login broken", "no user can sign in", "Sign-in is completely broken."),
    (3, "slow dashboard", "dashboard takes 30s for most users", "Dashboard is degraded for most users."),
    (3, "search flaky", "search times out for many queries", "Search is timing out frequently."),
    (3, "uploads degraded", "half of uploads fail and need retry", "Uploads fail about half the time."),
    (3, "emails delayed", "notification emails hours late for many", "Notifications are hours late."),
    (3, "api errors", "5% of api calls return 500", "A visible share of API calls error."),
    (3, "exports stuck", "many report exports never finish", "Exports hang for many users."),
    (2, "cannot export", "one account's csv export fails", "One customer cannot export data."),
    (2, "billing page error", "a single user sees an error on billing", "One user is blocked on billing."),
    (2, "password reset", "reset email never arrives for one user", "One user cannot reset a password."),
    (2, "import blocked", "one workspace's import always fails", "A single workspace cannot import."),
    (2, "2fa locked out", "one admin locked out of 2fa", "One admin is locked out."),
    (2, "invoice wrong", "one customer's invoice shows wrong total", "One invoice has a wrong total."),
    (1, "typo", "the word 'recieve' on the about page", "A typo with no functional impact."),
    (1, "color off", "button color differs from brand guide", "Cosmetic color mismatch only."),
    (1, "question", "how do I change my avatar", "A usage question, not a defect."),
    (1, "alignment", "footer misaligned on wide screens", "Minor layout issue only."),
    (1, "docs unclear", "setup guide step 3 is confusing", "Documentation clarity request."),
    (1, "feature ask", "please add dark mode", "A feature request, not an incident."),
    (0, "buy followers", "get 10000 followers cheap", "Spam with no relevance."),
    (0, "casino", "best online casino bonus click here", "Spam advertisement."),
    (0, "seo offer", "we will rank your site number one", "Unsolicited SEO spam."),
    (0, "crypto spam", "double your coins guaranteed", "Spam scam offer."),
    (0, "pills", "cheap pills no prescription", "Spam with no relevance."),
    (0, "lottery", "you won a prize claim now", "Phishing spam."),
    (4, "region outage", "eu region completely offline", "An entire region is offline."),
    (3, "webhooks lag", "webhooks delayed 20 minutes for many", "Webhooks are far behind for many."),
    (2, "sso broken for org", "one org's saml login fails", "A single org cannot log in."),
    (1, "tooltip typo", "tooltip says 'sucess'", "A tooltip typo only."),
    (0, "warranty call", "extended car warranty offer", "Irrelevant spam call transcript."),
    (2, "api key rotate", "one customer cannot rotate api key", "One customer blocked on key rotation."),
]

# input variants: the volume the v0.3 smoke run showed a 350M student needs
# before free-running JSON is reliably valid (~100 train rows)
VARIANTS = (
    ("", ""),
    ("re: ", " - reported via support"),
    ("fwd: ", " - escalated from chat"),
    ("ticket: ", " (second report today)"),
)


def triage_records(*, text: bool, decision: bool = True) -> list[dict]:
    records = []
    for prefix, suffix in VARIANTS:
        for priority, title, body, explanation in CASES:
            output: dict | int | str
            if text and decision:
                output = {"priority": priority, "explanation": explanation}
            elif text:
                output = explanation
            else:
                output = priority
            records.append(
                {
                    "input": {"title": f"{prefix}{title}", "body": f"{body}{suffix}"},
                    "output": output,
                }
            )
    return records


TRIAGE_PROMPT = (
    "Assign priority 0 (ignore) to 4 (page someone now) and explain in one "
    "short sentence.\n4: outage or data loss now\n3: degraded for many users\n"
    "2: single-user blocker\n1: cosmetic or question\n0: spam"
)


def lora_candidate(**overrides) -> dict:
    config = {
        "type": "lora",
        "model": BASE_MODEL,
        "precision": "auto",
        "max_epochs": 14,
        "patience": 4,
        "batch_size": 4,
        "eval_batch_size": 8,
        "max_seq_len": 512,
    }
    config.update(overrides)
    return config


def build_spec(name: str, output, *, candidate: dict | None = None, **extra):
    from smallbatch.spec import FunctionSpec

    return FunctionSpec(
        name=name,
        description="gpu regression",
        input_schema={"title": "string", "body": "string"},
        output=output,
        prompt=TRIAGE_PROMPT,
        candidates={"student": candidate or lora_candidate()},
        **extra,
    )


def compile_and_select(spec, records, tmp_path, *, cpu_threads=4):
    from smallbatch.api import compile as compile_fn
    from smallbatch.api import label, select
    from smallbatch.runtime import load_fn

    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, records, out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=cpu_threads)
    record = compiled.candidates["student"]
    assert record["status"] == "completed", record.get("error")
    selected = select(
        spec.name, "student", version=compiled.build_id,
        artifacts_root=root, interactive=False,
    )
    function = load_fn(spec.name, artifacts_root=root)
    return compiled, selected, function
