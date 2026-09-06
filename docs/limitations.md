# Limitations

- The committed provider is synthetic. Its findings cannot be transferred to real models.
- The task suite is intentionally compact and synthetic.
- Token counts are character-based estimates in the synthetic run; real APIs report provider token usage.
- External-provider adapters record usage but do not infer dollar cost from changing price lists.
- Lexical groundedness cannot determine factual truth or evaluate every valid paraphrase.
- The component library is not a fully balanced factorial design, so component-effect rows are descriptive.
- Repeated comparisons can produce false discoveries without multiplicity correction.
- A public holdout can eventually become a development set through repeated human iteration.
- Provider behaviour may change even when a moving model alias retains the same name.
- Prompt optimisation cannot repair missing knowledge, unsuitable models, weak retrieval or unsafe tool permissions.

These limitations are design constraints to manage, not footnotes to hide. A production study should extend the suite, use authorised domain data, calibrate graders against humans and preserve a genuinely unseen final test set.

