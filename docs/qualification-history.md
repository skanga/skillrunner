# Historical qualification evidence

The reference candidate (`0fb4ba6`) met the empirical qualification thresholds: 51 of 63 task trials (80.95%, including all failures and an independent grading correction) and 21 of 22 automatic-selection cases (95.45%, with one false activation). The task trials used a pinned GPT-5.5 executor with a Gemma image-inspection helper; these results do not guarantee the same quality from every compatible model.

Subsequent completion fixes added runtime-version provenance and precise endpoint-capability diagnostics. The empirical scores remain measurements of `0fb4ba6`; no new paid qualification was performed for those fixes or the usability remediation. Version probes add bounded process time and can stop a run if its deadline expires.

For reference candidate `0fb4ba6`, native CI passed on Linux, macOS, and Windows with Python 3.13 and 3.14. A separate Windows job passed the no-usable-POSIX-shell checks on product code identical to that reference candidate. Its tool-calling and artifact-output conformance passed against two distinct OpenAI-compatible endpoint implementations. See the [reference native CI run](https://github.com/skanga/skillrunner/actions/runs/36650728357) and [reference Windows shell qualification](https://github.com/skanga/skillrunner/actions/runs/36653125690). The subsequent completion fixes have [separate native CI checks](https://github.com/skanga/skillrunner/pull/23/checks).

Historical CI and benchmark results are not verification of later changes. Review the current tests and platform CI separately.
