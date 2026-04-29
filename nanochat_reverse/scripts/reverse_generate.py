"""
Generate with a reverse-trained base model.

The prompt is an ending anchor in normal reading order. Internally we reverse
the anchor tokens, generate more tokens in reverse order, then flip the whole
sample back so the printed text reads forward into the anchor.

Example:
    python -m scripts.reverse_generate --model-tag reverse_d24 --anchor "the answer is 42."
"""

import argparse

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


def main():
    parser = argparse.ArgumentParser(description="Generate lead-in text from a reverse-trained base model")
    parser.add_argument("--model-tag", type=str, default=None, help="Base checkpoint tag to load")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load; default is latest")
    parser.add_argument("--anchor", type=str, default="the end.", help="Ending anchor in normal forward text order")
    parser.add_argument("--max-tokens", type=int, default=128, help="Number of reverse-order tokens to generate")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="cuda|cpu|mps; empty autodetects")
    parser.add_argument("--show-reverse", action="store_true", help="also print the raw reverse-order model stream")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    anchor_ids = tokenizer.encode(args.anchor)
    reverse_prompt = [bos] + list(reversed(anchor_ids))
    top_k = None if args.top_k <= 0 else args.top_k

    samples, _ = engine.generate_batch(
        reverse_prompt,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
    )

    print(f"anchor: {args.anchor!r}")
    print()
    for i, sample in enumerate(samples, start=1):
        reverse_ids = sample[1:]  # drop BOS
        forward_ids = list(reversed(reverse_ids))
        forward_text = tokenizer.decode(forward_ids)
        print("=" * 80)
        print(f"sample {i}")
        print("-" * 80)
        print(forward_text)
        if args.show_reverse:
            print("-" * 80)
            print("raw reverse stream:")
            print(tokenizer.decode(reverse_ids))

    compute_cleanup()


if __name__ == "__main__":
    main()
