# Live experiment protocol

## Before execution

1. Define the use case and primary success metric.
2. Confirm that every example is public, synthetic or authorised for the selected provider.
3. Freeze the repository commit and exact task suite.
4. Select exact model versions where supported.
5. Set repetitions and decoding controls before seeing the final holdout.
6. Estimate maximum spend independently using current provider pricing.

## Execution

Use `promptlab run-live`. The command stores task-level output metrics locally and identifies the provider, model and prompt variant. It does not calculate dollar cost from a hard-coded price table because provider prices change independently of repository releases.

For cross-provider conclusions, use comparable capability tiers but avoid implying that differently priced or specialised models are identical treatments. Record API region, execution timestamp, SDK version, rate-limit behaviour and retry policy.

## Review and publication

- inspect raw failures and grader disagreements;
- remove or redact sensitive content;
- calculate confidence intervals across repeated trials;
- distinguish exploratory and confirmatory analyses;
- preserve negative results;
- publish exact model identifiers and task-set commit;
- include current pricing sources and calculation date;
- state whether humans reviewed the evaluation labels.

Never copy private enterprise prompts or client data into this public repository without explicit written authorisation.

