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

## Recommended Experiment Sequence

1. Train reverse-only 100M on raw web text.
2. Sample a fixed bank of ending anchors every validation checkpoint.
3. Compare repetition, coherence, and suffix adherence across checkpoints.
4. Fine-tune on QA/rationale examples with answer endings.
5. Generate multiple traces per fixed answer and score them with an external verifier.

## Important Failure Mode

The model can produce convincing but false explanations. Reverse generation answers:

```text
what text would make this ending likely?
```

It does not answer:

```text
what reasoning is true?
```
