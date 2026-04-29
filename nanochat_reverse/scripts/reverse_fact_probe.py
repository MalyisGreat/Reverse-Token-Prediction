"""
Probe a reverse-trained base model with factual ending anchors.

The anchors are normal forward text. The model sees the reversed anchor tokens,
generates earlier tokens in reverse order, and the printed sample is flipped
back into readable text.
"""

import argparse

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


HISTORY_ANCHORS = [
    "and in 1776, the Continental Congress adopted the Declaration of Independence.",
    "and in 1066, William the Conqueror defeated Harold Godwinson at the Battle of Hastings.",
    "and in 1453, Constantinople fell to the Ottoman Empire.",
    "and in 1865, the Thirteenth Amendment abolished slavery in the United States.",
    "and in 1914, the assassination of Archduke Franz Ferdinand helped trigger World War I.",
    "and on July 20, 1969, Neil Armstrong became the first person to walk on the Moon.",
    "and in 1989, the Berlin Wall fell.",
]


def generate_for_anchor(engine, tokenizer, anchor, args, seed):
    bos = tokenizer.get_bos_token_id()
    anchor_ids = tokenizer.encode(anchor)
    reverse_prompt = [bos] + list(reversed(anchor_ids))
    top_k = None if args.top_k <= 0 else args.top_k

    samples, _ = engine.generate_batch(
        reverse_prompt,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=seed,
    )

    print("=" * 100)
    print(f"anchor: {anchor}")
    print()
    for i, sample in enumerate(samples, start=1):
        reverse_ids = sample[1:]  # drop BOS
        forward_ids = list(reversed(reverse_ids))
        print(f"sample {i}")
        print("-" * 100)
        print(tokenizer.decode(forward_ids))
        print()


def main():
    parser = argparse.ArgumentParser(description="Run factual reverse-generation probes")
    parser.add_argument("--model-tag", type=str, default=None, help="Base checkpoint tag to load")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load; default is latest")
    parser.add_argument("--domain", type=str, default="history", choices=["history"])
    parser.add_argument("--anchor", action="append", default=[], help="Custom ending anchor; can be repeated")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="cuda|cpu|mps; empty autodetects")
    args = parser.parse_args()

    anchors = args.anchor if args.anchor else HISTORY_ANCHORS

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    try:
        model, tokenizer, _ = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        engine = Engine(model, tokenizer)

        print(f"domain: {args.domain}")
        print(f"anchors: {len(anchors)}")
        print(f"num_samples_per_anchor: {args.num_samples}")
        print()
        for i, anchor in enumerate(anchors):
            generate_for_anchor(engine, tokenizer, anchor, args, args.seed + i)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
