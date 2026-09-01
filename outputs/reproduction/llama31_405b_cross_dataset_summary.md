# Llama 3.1 405B Second Chance cross-dataset summary

Model: `llama-3.1-405b-instruct`  
Checkpoint: `RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic`  
Serving: vLLM on 8 x A100 SXM4 80 GB, temperature 0, 20 top logprobs

| Dataset | Usable | Baseline accuracy | Game switch | Neutral switch | Lift p | Changed AccIncor | Second choice | Game entropy minus baseline | Tests passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| SimpleMC | 495/500 | 59.8% | 16.0% | 11.1% | 0.000536 | 48/63 (76.2%) | 53/79 (67.1%) | -0.039 bits | 4/4 |
| TriviaMC | 495/500 | 81.0% | 9.3% | 5.9% | 0.00599 | 29/38 (76.3%) | 34/46 (73.9%) | -0.013 bits | 4/4 |

AccIncor uses the paper-aligned analysis conditional on an initially incorrect
answer and an answer change, tested against 1/3. Second-choice rates and entropy
use the paper tests; the reports also contain coverage-robust A-D diagnostics.

The TriviaMC batch reused the preserved checkpoint and incurred approximately
$1.04 in Vast compute and running-disk charges. The instance was stopped after
the artifacts were retrieved.

Detailed reports:

- `outputs/reproduction/simplemc_llama31_405b/REPLICATION_REPORT.md`
- `outputs/reproduction/triviamc_llama31_405b/REPLICATION_REPORT.md`
