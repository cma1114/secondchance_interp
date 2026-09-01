# Candidate models for the next Second Chance interpretability study

Research date: 2026-08-02

## Completed SimpleMC screens

The first candidate screen is complete; see `outputs/reproduction/SIMPLEMC_CANDIDATE_COMPARISON.md` for the full report.

- **Qwen3.5-122B-A10B:** passes Lift, changed-trial AccIncor, and SecChoice, but fails NoEntInc. Game switching was 34.8% versus 15.6% neutral (paired p=4.55e-16); Game-minus-baseline entropy was +0.301 bits. This is a stronger-capability instance of the established Qwen three-of-four profile, not an all-four pass.
- **Gemma 4 26B-A4B IT:** fails Lift despite passing the other three tests. Game switching was 32.3% versus 30.9% neutral (p=0.392). Its large baseline-to-redo probability redistribution is almost identical in Game and neutral, making it a useful generic-redo negative control but not a candidate for the feedback-specific mechanism.

The Qwen result keeps Qwen3.5-122B in consideration if the goal is to study the family-level strategic compression/flattening signature at greater capability. Gemma 4 26B-A4B should not be the primary model organism for that mechanism.

This document supersedes the preliminary shortlist. Model-specific interpretability artifacts are an advantage, not an entry requirement. Selection should happen in two stages:

1. Screen capable, current open-weight models that can produce an immediate, non-reasoning answer for behavioral success in the Second Chance Game.
2. Among the models that actually pass, prefer tractable architectures and existing model-specific interpretability artifacts.

Reversing these stages would systematically eliminate the largest and most capable candidates before testing the phenomenon.

## Behavioral target

On SimpleMC, the ideal model passes all four paper tests: Lift, change-conditioned AccIncor against 1/3, SecChoice against 1/3, and NoEntInc. The fallback passes the first three and fails only NoEntInc.

The paper's consistent full passes were closed OpenAI models. Qwen 3, Grok 3, and Gemini 2.5 Flash non-thinking passed the first three but increased entropy. Our Qwen replications reproduce that three-of-four profile. There is no published Second Chance evaluation of the open-weight candidates below, so behavior must be screened empirically.

## Large, current behavioral candidates

### 1. Qwen3.5-122B-A10B and Qwen3.5-397B-A17B

These are the most important untested large-model candidates. Both are current open-weight MoEs, and both officially support an instruct/non-thinking mode through `enable_thinking=False`. The 122B model activates roughly 10B parameters per token; the 397B model roughly 17B. Total parameter count therefore overstates their per-token compute, but their capability is the relevant reason to screen them.

They do not currently have the same turnkey interpretability stack as the smaller candidates. Qwen-Scope's released models top out at Qwen3.5-35B-A3B, and Qwen3.5 combines MoE routing with Gated DeltaNet/attention hybrid blocks. Current J-Lens support does not cover that hybrid architecture. If one passes all four behavioral tests, however, that result is more important than this inconvenience: ordinary residual-stream hooks, logit-lens/probe analysis, attention-block analysis, router analysis, and task-specific learned probes remain possible.

The 122B model is the sensible first large screen because it is much cheaper to serve. The 397B model is the stronger capability test if the 122B result is negative or ambiguous.

Sources: [Qwen3.5-122B model card and non-thinking configuration](https://huggingface.co/Qwen/Qwen3.5-122B-A10B), [Qwen3.5 architecture announcement](https://qwen.ai/blog?id=qwen3.5), [released Qwen-Scope models](https://huggingface.co/models?other=qwen-scope).

### 2. Qwen3-235B-A22B-Instruct-2507 — established scale anchor, not a new replacement

This is already our large-model reference. It supports immediate non-thinking answering and, in our replications, passes Lift, change-conditioned AccIncor, and SecChoice while failing NoEntInc across the tested datasets. It should remain in every cross-model comparison, but rerunning it does not answer the question of whether a different open model can pass all four tests.

Its 235B-total/22B-active MoE architecture is also substantially less convenient than a dense model for component-level interpretation, and no comparably mature NLA/AO/J-Lens or every-layer SAE stack was found for this exact checkpoint.

### 3. Gemma 4 31B IT — not larger in parameter count, but a much newer capability candidate

Gemma 4 31B is a 31B **dense** transformer, not an MoE. Google reports configurable thinking, including an off mode, although the exact empty-thought chat-template behavior must be pinned and verified in the logged prompt. It is worth screening despite being one billion parameters smaller than Qwen3-32B because it is a new generation with much stronger reported capabilities than Gemma 3.

It already has a public 131k Matryoshka residual-stream SAE at layer 30 and Neuronpedia HeadVis support. That is not the every-layer Gemma Scope 2/NLA/AO stack available for Gemma 3, but it is enough to make a successful behavioral result immediately actionable.

Sources: [official Gemma 4 overview](https://ai.google.dev/gemma/docs/core), [official prompt/thinking format](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4), [Gemma 4 31B SAE and HeadVis](https://www.neuronpedia.org/gemma-4-31b).

## Artifact-rich primary candidates

### Gemma 3 27B IT — strongest interpretability ecosystem

Gemma 3 27B is currently the best-supported serious model organism for this project.

Available artifacts:

- **Gemma Scope 2:** SAEs and transcoders trained on every layer of Gemma 3, including skip-transcoders and cross-layer transcoders intended for tracing multi-step computation. Features can be explored and steered through Neuronpedia.
- **Natural Language Autoencoder:** Anthropic released an activation verbalizer and activation reconstructor for layer 41 of 62 (`kitft/nla-gemma3-27b-L41-av` and `-ar`).
- **Activation Oracle:** a released Gemma-3-27B checkpoint exists in the Activation Oracles collection.
- **Conventional tooling:** Hugging Face hooks, standard logit/probe analysis, attention inspection, and Jacobian-lens-style analysis are all feasible. Gemma's interleaved local/global attention is an added detail, not an opaque recurrent state like Gated DeltaNet.

Behavioral uncertainty is the major drawback. Google-family performance in the paper supplies only a weak prior, and Gemma is not Gemini. Nevertheless, if it passes, it gives us several independent ways to ask whether the incorrect-feedback instruction is represented as “the previous candidate must be suppressed,” generic uncertainty, or something else.

Behavioral API caveat: OpenRouter currently exposes top logprobs for Gemma 3 27B through an FP8 Parasail endpoint, while the available BF16 endpoint does not expose them. Any candidate result should therefore be reproduced self-hosted in BF16 before small mechanistic effects are interpreted.

Sources: [Gemma Scope 2](https://deepmind.google/models/gemma/gemma-scope/), [NLA release and checkpoints](https://github.com/kitft/natural_language_autoencoders), [Activation Oracle collection](https://huggingface.co/collections/adamkarvonen/activation-oracles), [Gemma 3 model card](https://huggingface.co/google/gemma-3-27b-it).

### Qwen3-32B — strongest behavioral prior plus substantial tooling

Qwen3-32B is the strongest candidate for reproducing the behavior while eliminating Qwen3.6's Gated DeltaNet complication.

Available artifacts:

- **Activation Oracle:** a released Qwen3-32B checkpoint exists, although it is not the flagship/best-validated Qwen3-8B oracle and should be independently controlled for hallucination and text inversion.
- **Sparse autoencoders:** BatchTopK/JumpReLU-compatible residual-stream SAEs at 25%, 50%, and 75% depth, with 16k and 65k dictionaries and multiple sparsity targets.
- **Neuronpedia:** hosted Qwen3-32B SAE features and an attention-head visualizer.
- **J-Lens:** Qwen3's conventional dense decoder is directly supported; unlike a raw logit lens, the Jacobian lens estimates how an intermediate activation will be transported to the final residual state.
- **Ordinary architecture:** 64 dense full-attention transformer layers, no recurrent/linear-attention blocks and no MoE routing.

Limitations:

- The released SAEs cover three depths, not every layer. Their trainer explicitly filtered rare extreme-norm Qwen activations, which must be handled consistently in our prompts.
- No released Anthropic NLA exists for Qwen3-32B.
- The OpenRouter provider exposing top logprobs is Groq and does not identify numerical precision; a self-hosted BF16 reproduction is required.

Sources: [Qwen3-32B SAEs](https://huggingface.co/adamkarvonen/qwen3-32b-saes), [Neuronpedia](https://www.neuronpedia.org/qwen3-32b), [Activation Oracle collection](https://huggingface.co/collections/adamkarvonen/activation-oracles), [J-Lens documentation](https://rapidmlx.com/docs/jlens), [Qwen3 configuration](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json).

## Useful scaling controls

### Gemma 3 12B IT

Gemma Scope 2 provides every-layer SAEs/transcoders and Anthropic released a layer-32 NLA. It is much cheaper to run than the 27B model, but has a weaker capability prior and no released AO in the current collection. It is valuable if the 27B model succeeds and we want a within-family emergence comparison.

### Qwen3-14B and Qwen3-8B

Both have standard dense architectures, released SAEs, released Activation Oracles, and J-Lens compatibility. Qwen3-8B has the flagship/best-documented open AO. They are useful scaling controls but less likely than Qwen3-32B to pass AccIncor and SecChoice on difficult items.

### Llama 3.3 70B Instruct

Llama 3.3 is old as a behavioral candidate, but it is unusually well provisioned for learned interpretability: Anthropic released both a layer-53 NLA and a relatively fully trained AO. It remains a methodological control, not a leading behavioral bet.

## Deprioritized candidates

- **GPT-OSS-20B/120B:** Harmony reasoning cannot be cleanly disabled, so the final answer follows a variable generated reasoning trace. This is not the immediate-answer paradigm and is a nonstarter.
- **Mistral models:** clean blocks alone do not compensate for weak behavioral priors or the absence of comparable model-specific interpretability artifacts.
- **Gemma 4 31B:** no longer deprioritized. It is dense, current, and already has a public mid-layer SAE; it belongs in the behavioral screen even though its learned-tool stack is thinner than Gemma 3's.
- **Qwen3.5/Qwen3.6 hybrids:** Qwen-Scope provides useful SAEs, but linear-attention/Gated DeltaNet state remains precisely the complication we are trying to escape; current J-Lens support explicitly excludes hybrid linear-attention models.
- **Other frontier-scale open MoEs (Llama 4, DeepSeek, Kimi, GLM):** not ruled out by size or architecture, but placed behind the Qwen3.5 large models because they combine weaker model-specific interpretability support with less direct continuity to our replicated Qwen result. They become worthwhile if the first large screens fail.

## What the additional tools could contribute

These techniques answer different questions and should not be conflated:

- **J-Lens:** a better readout of when each answer candidate and the “previous answer is invalid” constraint become consequential for the final state. It is still observational.
- **NLA:** at its trained layer, ask for an unconstrained description of the Game-minus-baseline or Game-minus-neutral activation. This is useful for hypothesis generation, but descriptions can confabulate and must be checked against the reconstructor and text-only controls.
- **Activation Oracle:** directly query whether an activation represents the prior answer, an instruction to avoid it, uncertainty, or a plan to select another answer. It is flexible but non-mechanistic and vulnerable to context reconstruction/text inversion; shuffled-activation, zero-activation, and prompt-only controls are mandatory.
- **SAEs:** search for sparse features selectively activated by incorrect feedback, by compression of the A-D distribution, or by switching. Feature activation can be related to answer geometry across trials.
- **Transcoders/attribution graphs:** for Gemma 3, trace which sparse features and component writes produce the final loss of baseline-winner advantage. This is the closest of the available tools to a component-level mechanism.

An NLA or AO output would not itself establish a mechanism. Its value is that it can generate a specific, human-readable hypothesis that can then be checked with the SAE/transcoder decomposition and simple interventions.

## Recommended decision

Use a staged screen rather than selecting only from artifact-rich models. First run a sufficiently powered behavioral screen on **Qwen3.5-122B-A10B, Gemma 4 31B IT, Gemma 3 27B IT, and Qwen3-32B** using the existing frozen questions and exact paper tests. Escalate promising or borderline results to the full 500 questions. If Qwen3.5-122B is negative or ambiguous, add **Qwen3.5-397B-A17B**. Keep the existing Qwen3-235B result as the large-scale anchor rather than treating it as a new candidate.

Pin the provider and exact non-thinking chat template, request the maximum useful top-logprob coverage, aggregate A-D token variants, and retain raw probability diagnostics.

- If **Gemma 3 27B** passes, prefer it because it has the richest complementary interpretability stack.
- If **Qwen3-32B** passes and Gemma does not, use Qwen: it retains the strongest family-level behavioral prior, offers AO/SAE/J-Lens access, and removes Gated DeltaNet.
- If a **large Qwen3.5** model uniquely passes all four tests, accept the harder architecture and begin with the simple residual/probe analyses already implemented; do not discard the behavioral phenomenon merely because pretrained interpretability artifacts are absent.
- If both pass, the pair is scientifically stronger than either alone: one offers every-layer sparse circuit tools and NLAs, while the other connects directly to the already-replicated Qwen behavioral signature.
- If neither passes, test Gemma 3 12B and Qwen3-14B only as scaling diagnostics, not as likely replacements.

Paper source: [Evidence for Limited Metacognition in LLMs](https://arxiv.org/abs/2509.21545).
