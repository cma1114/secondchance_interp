# GitHub payload workflow

This working directory contains reproducible source code alongside several
gigabytes of raw activations, checkpoints, logs, caches, and superseded
intermediate results. Git tracking is therefore source-first and allowlisted.

The policy is `scripts/github_payload_policy.json`. The audit tool is
`scripts/audit_github_payload.py`.

The intended initial-publication workflow is:

```bash
python scripts/audit_github_payload.py --write-manifests
git init
python scripts/audit_github_payload.py --stage --check-index
git status --short
```

The tool includes source, configs, tests, frozen root datasets, compiled
behavioral results, and small canonical artifacts linked from `README.md`.
It refuses credential paths, secret-looking literals, raw tensor/checkpoint
formats, symlinks, and individual files larger than 5 MiB. It stages exact
paths and never uses `git add .`.

`github_payload_manifest.json` records the exact included files with sizes and
SHA-256 hashes. `excluded_artifacts.json` summarizes omitted local material and
hashes every omitted file of at least 5 MiB, without embedding any artifact
content. Raw artifacts should later be placed in a dedicated scientific
archive if exact third-party reanalysis requires them; they should not be put
in Git LFS merely to make an initial GitHub push possible.

After staging, inspect the audit output and `git status` before committing.
Creating a remote and pushing are deliberately separate actions.

For later updates, rerun the same audit with `--write-manifests --stage
--check-index`. The staging step adds newly eligible paths and removes paths
that no longer qualify from Git's index without deleting their local files.
