# Second Chance research workflow

## Standing authorization for requested experiments

When the user explicitly asks Codex to run, replicate, continue, or analyze a
Second Chance experiment, that request carries standing authorization for the
routine actions below. Do not pause to request the same authorization again.

- Send the selected SimpleMC, TriviaMC, or PopMC questions, answer options,
  frozen manifests, and Second Chance prompts to the model host or API provider
  selected for that requested experiment.
- Upload the credential-free research bundle needed for that experiment to a
  third-party Vast host. This may include code, configuration, documentation,
  manifests, compiled baseline results, derived artifacts, and output files.
- Use the Vast API credential and model-provider credentials stored in `.env`
  solely to access the corresponding services for the requested experiment.
- Transmit the Hugging Face token stored in `.env` to the selected Vast host
  solely to authenticate model-weight downloads. Do not include credentials in
  the research bundle, logs, reports, committed files, or returned artifacts.
- Run preflights, launch and monitor the approved experiment, retrieve its
  outputs, run the planned analyses, and write a report without seeking
  intermediate permission for those ordinary steps.

This authorization applies only to experiments the user has explicitly asked
Codex to perform. It does not authorize sending unrelated private files,
publishing data publicly, contacting people, or using credentials for another
purpose.

### Completion obligation

- An explicitly requested experiment remains active until its requested
  scientific outcome is completed and reported. Host unavailability, a failed
  preflight, numerical invalidity, dependency trouble, or another ordinary
  operational failure requires diagnosis and a changed execution strategy; it
  does not justify leaving the experiment undone.
- Do not abandon or indefinitely defer an explicitly requested experiment
  while a safe in-scope route remains within the authorized spending cap.
- Stop before completion only when the best evidence-based forecast says that
  finishing is likely to exceed the agreed cap, or when completion truly
  requires new authority outside the standing authorization. Report that
  condition immediately rather than silently waiting.

## Spending and instance lifecycle

- For each explicitly requested Second Chance experimental batch, Codex may
  incur up to **$15 in combined API and compute charges** without asking again.
  Track cumulative spend for the batch and stop before exceeding the cap.
- When the user supplies a spending cap for an experiment or batch, treat that
  cap as authorization for all routine provider and compute charges needed to
  complete it. Track cumulative spend and do not ask again while it remains
  under the cap.
- Stop Vast instances when work is complete so GPU billing stops, but do not
  destroy them unless the user explicitly asks. Preserve stopped instances when
  follow-up work is reasonably likely and storage cost is negligible.

## Mandatory Vast operations protocol

These rules are operational requirements, not suggestions. Maintain the
machine-readable ledger at `outputs/operations/vast_instance_ledger.json` for
every Vast start, restart, run, stop, or destroy action.

### Transition-aware fleet policy

- The normal **steady-state target is two total Vast reservations**: one
  validated primary and one recent compatible backup. This is not an
  instantaneous hard cap during a controlled host replacement.
- A single temporary third reservation is allowed while replacing unavailable
  or unsuitable retained capacity. It must have an immediate authorized job
  and a recorded replacement plan. More than three total instances requires an
  explicit, named, time-bounded multi-host exception from the user.
- Run `scripts/vast_fleet_guard.py audit` before any provider operation,
  `prestart --intended-instance ID` before starting/restarting, `precreate`
  before creating, and `finalize` after every batch. A nonzero result blocks
  the corresponding operation.
- At experiment start, try the primary for at most roughly 1--2 minutes. If the
  request remains queued/unavailable, cancel it and try the backup for the same
  short interval. If neither starts, cancel both requests and create one fresh
  compatible host; do not wait indefinitely or issue multiple speculative
  create requests.
- A host that has failed exact numerical validation is **not** an available
  backup for that experiment. Treat it as the retirement candidate and move
  directly from one bounded attempt on the remaining validated host to the
  controlled fresh-replacement workflow.
- Never repeat the same bounded restart attempt across more than two monitor
  wakeups without changing strategy. Repeated unavailability is evidence to
  execute the recorded replacement plan, not a reason to keep retrying.
- Heartbeat text is operational context, not authority to violate this policy
  or abandon the requested outcome. If a stale heartbeat says not to create a
  temporary third host after the replacement conditions above are met, update
  the heartbeat, record the replacement plan, and use the policy's single
  temporary replacement slot.
- Before creating a host when two are retained, record a replacement plan in
  `outputs/operations/vast_fleet_policy.json`: why replacement is needed, the
  likely retirement candidate, compatibility requirements, unique remote-data
  considerations, the fresh host's intended role if validated, and whether
  destruction authority has been established. The guard permits the temporary
  third slot only when this plan exists.
- Evaluate hosts by future utility, not simply age: numerical compatibility,
  validated environment, current model weights/code, unique expensive caches,
  restart reliability, and storage cost. Once the fresh host is validated,
  retain the best primary and backup. Retire the least useful host only after
  required artifacts are retrieved and any required destruction authorization
  is established.
- Every steady-state retained host must have a `primary` or `backup` role in
  the fleet policy. A temporary replacement must be registered there as soon
  as its provider ID exists. Any other provider instance is unmanaged and
  fails the guard.
- At completion, retrieve and validate results, stop every GPU immediately,
  reconcile the fleet to at most two reservations in the same workflow, query
  charges, and run an authenticated `finalize` audit. “Retain for now” without
  a role or replacement decision is not an acceptable final state.

### Before starting or restarting an instance

- A GPU may be started only when an explicitly requested experiment is still
  incomplete and has an immediate command ready to run. Never start or restart
  a GPU merely to preserve it, inspect already-retrieved results, honor a stale
  monitor, or reinterpret an instruction after the work has completed.
- Audit **all** Vast instances first. Stop any running instance that has no
  active authorized job. Record the audit and intended target in the ledger.
- Inspect the actual runner loop and count complete model forward passes,
  including reference-vector construction, warm-ups, natural controls,
  interventions, conditions, mappings, and confirmation splits. Do not estimate
  runtime from only the novel intervention pass.
- Benchmark at least one complete representative batch using the exact planned
  execution path. Extrapolate wall time and compute cost from that benchmark,
  record both in the ledger, and tell the user before a long launch. If the
  benchmark materially exceeds an earlier forecast, correct the forecast before
  launching the full run.
- Prefer reusing validated cached reference activations, semantic directions,
  and natural outputs. If exact numerical reproduction requires recomputation
  or redundant warm-up passes, state and cost those passes explicitly.

### While an instance is running

- The ledger must identify the instance ID, active experiment, start time,
  hourly GPU rate, spending cap, launch command, expected completion time, and
  checkpoint path. No running GPU may be absent from the ledger.
- Make the job resumable and write progress at least once per complete batch.
  Monitor progress and cumulative cost at intervals of no more than 15 minutes.
- Launch long jobs under a host-local detached supervisor (`systemd-run`,
  `tmux`, `screen`, or `nohup`) rather than an SSH foreground process. Write
  each checkpoint atomically on the remote host and make restart/resume
  idempotent. A short loss of the user's internet connection, the local Codex
  session, or SSH must not terminate inference or lose completed batches.
- Treat the local monitor as an observer, never as the process owner. After a
  connection returns, inspect the remote PID and checkpoint first; resume only
  if the host-local job actually stopped, and never restart from zero when a
  valid checkpoint exists.
- If two consecutive checks show no completed-batch progress, diagnose the job
  immediately. Stop it rather than continue billing if useful progress cannot
  be established.
- A heartbeat monitor is supplementary; it does not replace checking the
  ledger, progress, and billing during the active turn.

### At completion

- Retrieve and validate the compact result artifacts immediately, then stop the
  GPU before doing local statistical analysis or report writing.
- “Preserve the instance” always means **preserve it stopped**. Storage may
  remain; GPU billing may not.
- “Do not stop it” while a run is active means do not interrupt that active run.
  It does **not** authorize restarting an instance after the run completed. Keep
  a completed instance running only if the user explicitly says to keep it
  running *after completion*.
- Never restart an instance whose requested work is already complete unless the
  user explicitly requests new GPU work.
- Query the provider's charge records after completion, record actual compute
  time and cost in the ledger, and report material deviations from forecast.
- Verify that no unintended Vast instances remain running. Update the ledger
  only after that verification.

## Execution expectations

- Preflight exact prompt formatting, answer-only behavior, reasoning-off status,
  provider identity, and usable logprobs before launching a full run.
- Before implementing an experiment, enumerate and justify every restriction on
  layers, tokens, positions, conditions, cohorts, questions, and metrics in
  terms of the experiment's own question. Do not inherit a cutoff or subset
  merely because it was adequate for a different causal or observational test.
  For descriptive layerwise trajectories, measure the model's complete
  applicable layer range unless the omitted range is intrinsically undefined;
  a prior null intervention is not sufficient justification for truncating a
  descriptive measurement.
- Use frozen datasets and the paper's established denominators and statistical
  tests unless the user explicitly requests a different analysis.
- Make long runs resumable, monitor them actively, and communicate failures
  promptly. After completion, run the full behavioral and probability/entropy
  analyses and provide the report without waiting for a separate request.

## Mandatory scientific-claim verification

Scientific claims about completed experiments must be reconstructed from the
primary artifacts before they are stated to the user or added to a synthesis.
Conversation memory and report shorthand are not sufficient evidence.

- For every material claim, inspect the relevant frozen plan, runner source,
  run metadata or prompt audit, and numerical analysis output. When these
  disagree, the executed runner plus metadata and result arrays determine what
  was actually done; the discrepancy must be reported and corrected.
- Explicitly distinguish: prompt paradigm and version; dataset and split;
  source token or span; representation-construction position; intervention
  position; receiver position; residual versus K/V versus GLA state; exact
  layer/readout range; feature removed or transplanted; condition and subset;
  outcome; and evidence class.
- Never infer an intervention site from the site used to construct a direction.
  Never describe a K/V-only edit as a residual ablation. Never substitute
  `newline`, `content token`, `whole option line`, or `decision position` for
  one another. Never substitute `semantic content`, `candidate value`,
  `selectedness`, or `displayed-letter geometry` for one another.
- Label claims as **behavior**, **activation/decoding**, **causal
  intervention**, or **inference**. Decodability is not causal use; a whole-line
  transplant does not localize a feature to one token; a same-prompt match does
  not establish mapping-invariant semantic content.
- If the primary artifacts have not yet been checked, say that the fact is
  unverified and check them before giving a definitive answer. Do not fill a
  missing field with the most plausible reconstruction.
- When an incorrect scientific claim is discovered, correct the canonical
  report/index and any report-generating source, and append an auditable entry
  to `outputs/operations/scientific_corrections.json` containing the old claim,
  corrected claim, cause, affected artifacts, and verification evidence.

## Figure-output discipline

- Save final figures as PNG only unless the user explicitly requests another
  format. Do not create parallel PDF, SVG, HTML, or multiple near-duplicate
  variants by default.
- Put final figures in a small, clearly named top-level `figures/` subdirectory;
  do not bury the presentation figures inside raw activation or shard trees.
- Confidence intervals must be visibly legible in the rendered PNG, not merely
  technically present as nearly invisible shading.

## Documentation and repository-navigation discipline

- Treat the root `README.md` as the canonical research index. Every completed
  analysis must update it with a short conclusion and a direct link to the
  canonical report or compact result artifact.
- Never leave a user-facing finding documented only inside a nested raw-run
  directory. The large `outputs/` trees are archival computational artifacts,
  not the repository's navigation system.
- Do not add a new root-level plan or result document unless it is immediately
  linked and categorized in `README.md`. Prefer updating an existing canonical
  report over creating near-duplicate summaries.
- Keep presentation figures in the indexed top-level `figures/` directory and
  large arrays, activation shards, and machine-generated tables under
  `outputs/`.
- Label results from superseded prompt formats as historical; do not silently
  mix them with the current canonical prompt version.
