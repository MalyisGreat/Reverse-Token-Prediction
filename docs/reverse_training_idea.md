# Reverse Training Idea

## What We Are Testing

The main question is whether a language model trained backward can produce useful text that explains or leads into a known ending.

Forward LMs are good at:

```text
given the beginning, predict what comes next
```

Reverse LMs test:

```text
given the ending, predict what came before
```

That makes the model a prequel generator: it can generate plausible preceding context for a suffix.

## Why This Might Be Interesting

For ordinary web text, reverse generation can create paragraphs that lead into an ending sentence. For QA/rationale data, it could become an answer-to-rationale generator:

```text
Therefore, the answer is Paris.
```

The model may generate:

```text
Question: What is the capital of France?
Reasoning: France's capital city is Paris.
```

This is not guaranteed reasoning. It is abductive generation: the model invents a plausible path to a fixed ending. A verifier is needed if the trace is supposed to be true.

## Baseline Result From The Local 53M Run

The current local 53.6M reverse-only model was trained on FineWeb-Edu-style data. After about 1.5B tokens, reverse validation reached roughly:

```text
reverse_loss ~= 2.9469
ppl ~= 19.05
```

The model learned backward local structure but still produced repetition, citation fragments, and weak behavior for short anchors.

## Why The H100 100M Run Is The Next Step

A roughly 100M parameter model should improve:

- sentence-level coherence
- topic stability before the suffix
- handling of longer ending anchors
- fewer broken starts
- less repetition, though not eliminating it

It will probably not become a normal chatbot without instruction or rationale fine-tuning.

## H100 Architecture Update

The H100 trainer now includes the strongest low-risk ideas from Gemma 3/4 and DeepSeek V3/V4 while keeping the experiment a reverse language model:

- grouped-query attention with `n_heads=12` and `n_kv_heads=4`
- hybrid local/global attention
- a 5:1 local-to-global layer pattern with the final layer always global
- local sliding windows defaulting to 512 tokens
- optional FlashAttention backend for true fast local-window attention
- optional sequence-length curriculum via `--seq_len_schedule`
- optional reverse MTP control through the `reverse_mtp2_low` experiment

The default H100 launch uses `seq_len=1024`, `batch_size=128`, `d_model=768`, `n_layers=12`, and `ffn_hidden=2560`, which keeps the model near the 100M parameter target after GQA removes KV projection weights.

For a new H100 box, run `scripts/bench_h100_attention.sh` first. If FlashAttention is not installed, PyTorch SDPA may make the local-window variant slower than global attention even though it is architecturally better for long-context scaling.

## Budget RunPod Path

For limited rented compute, use `scripts/runpod_h100_speedrun.sh`. It does three things before the real run:

- installs FlashAttention when possible
- tokenizes FineWeb-Edu once into a local binary token file
- benchmarks attention variants before launching the long run

The token binary matters because streaming Hugging Face data and tokenizing text inside the dataloader can leave an H100 underfed. The default speedrun target is 1B training tokens, with a 10% token-buffer margin for validation and sampling.

## Recommended Experiment Sequence

1. Run the H100 attention benchmark.
2. Train reverse-only 100M on raw web text with the fastest stable attention profile.
3. Sample a fixed bank of ending anchors every validation checkpoint.
4. Compare repetition, coherence, and suffix adherence across checkpoints.
5. Optionally run `EXPERIMENT=reverse_mtp2_low` as a small auxiliary-loss ablation.
6. Later, fine-tune on QA/rationale examples with answer endings and validate traces separately.

## Important Failure Mode

The model can produce convincing but false explanations. Reverse generation answers:

```text
what text would make this ending likely?
```

It does not answer:

```text
what reasoning is true?
```
