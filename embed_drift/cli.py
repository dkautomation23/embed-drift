# -*- coding: utf-8 -*-
"""The command line: take a fingerprint now, compare against it later.

    py -m embed_drift.cli baseline --model all-minilm --out fp.json
    py -m embed_drift.cli check fp.json --model all-minilm
    py -m embed_drift.cli check fp.json --model nomic-embed-text

Exit codes are the point in CI: 0 when the space is where you left it, 1 when it
moved, 2 when the question could not be asked at all (endpoint down, wrong probe
set, dimension change). A check that cannot run must not look like a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .geometry import compare
from .probes import PROBES, probe_hash
from .provider import ProviderError, embed_all

# Thresholds are defaults, not truths. A same-model re-run on Ollama sits at
# direct 1.0 and geometry 1.0, so anything meaningfully below is a real change;
# 0.99 leaves room for float noise and for a provider that rounds differently.
DIRECT_FLOOR = 0.99
GEOMETRY_FLOOR = 0.99


def _embed(args) -> list[list[float]]:
    return embed_all(
        PROBES,
        model=args.model,
        base_url=args.base_url,
        api=args.api,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
    )


def _baseline(args) -> int:
    vectors = _embed(args)
    fingerprint = {
        "model": args.model,
        "api": args.api,
        "base_url": args.base_url,
        "dimension": len(vectors[0]),
        "probe_hash": probe_hash(),
        "probe_count": len(PROBES),
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vectors": vectors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    print(
        f"fingerprint written: {args.out}\n"
        f"  model      {args.model}\n"
        f"  dimension  {len(vectors[0])}\n"
        f"  probes     {len(PROBES)} (hash {fingerprint['probe_hash']})"
    )
    return 0


def _check(args) -> int:
    # A missing or unreadable fingerprint is an ordinary situation - a first run,
    # a wrong path, a half-written file - and deserves a sentence, not a stack
    # trace. It is also not a pass, so it exits 2.
    try:
        saved = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"no fingerprint at {args.fingerprint}\n"
            "  Take one first:  embed-drift baseline --model <name> --out "
            f"{args.fingerprint}",
            file=sys.stderr,
        )
        return 2
    except (json.JSONDecodeError, OSError) as error:
        print(f"cannot read {args.fingerprint}: {error}", file=sys.stderr)
        return 2

    # Refuse rather than report a meaningless number.
    if saved.get("probe_hash") != probe_hash():
        print(
            "cannot compare: this fingerprint was taken with a different probe set\n"
            f"  fingerprint {saved.get('probe_hash')}\n"
            f"  this build  {probe_hash()}",
            file=sys.stderr,
        )
        return 2

    model = args.model or saved["model"]
    args.model = model
    try:
        vectors = _embed(args)
    except ProviderError as error:
        print(f"cannot compare: {error}", file=sys.stderr)
        return 2

    if len(vectors[0]) != saved["dimension"]:
        print(
            f"DRIFTED - dimension changed: {saved['dimension']} -> {len(vectors[0])}\n"
            "  Nothing in the old index is comparable to anything new. Reindex.",
            file=sys.stderr,
        )
        return 1

    result = compare(saved["vectors"], vectors)
    moved = result["direct_mean"] < DIRECT_FLOOR
    reshaped = result["geometry"] < GEOMETRY_FLOOR

    print(f"baseline   {saved['model']}  taken {saved['taken_at']}")
    print(f"now        {model}")
    print(f"dimension  {saved['dimension']}")
    print(f"direct     mean {result['direct_mean']:.4f}  min {result['direct_min']:.4f}")
    print(f"geometry   {result['geometry']:.4f}")
    if result["worst_probe"] >= 0:
        worst = PROBES[result["worst_probe"]].replace("\n", " ")[:60]
        print(f"worst probe #{result['worst_probe']}: {worst!r}")

    if not moved and not reshaped:
        print("\nOK - the space is where you left it.")
        return 0

    if moved and not reshaped:
        print(
            "\nDRIFTED - the encoding changed but the shape survived.\n"
            "  Vectors moved while the distances between probes did not, which is what a\n"
            "  renormalisation or a rescaling looks like. Your old vectors are no longer\n"
            "  comparable to new ones, so reindex - but retrieval quality itself is\n"
            "  probably intact once you do."
        )
        return 1

    print(
        "\nDRIFTED - the shape changed. This is the bad one.\n"
        "  Which probes are near which has been rearranged, so the neighbours your index\n"
        "  returns are no longer the neighbours it was built to return. Reindex, then\n"
        "  re-measure retrieval quality rather than assuming it came back."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    # Printing must not be able to fail the tool. A Windows console is not UTF-8
    # by default, and one unusual character in a message turned a successful
    # check into a UnicodeEncodeError after all the work was already done.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="embed-drift", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model", help="embedding model name")
        p.add_argument("--base-url", default="http://127.0.0.1:11434")
        p.add_argument("--api", default="ollama", choices=["ollama", "openai"])
        p.add_argument(
            "--timeout",
            type=float,
            default=600.0,
            # A cold model has to load before it can embed, and on a shared
            # machine that can mean minutes. Ten of them, because a false
            # "endpoint down" is worse than waiting.
            help="seconds per request (default 600)",
        )
        p.add_argument(
            "--api-key-env",
            help="name of the environment variable holding the key; the value is never printed",
        )

    base = sub.add_parser("baseline", help="take a fingerprint of the current model")
    common(base)
    base.add_argument("--out", type=Path, required=True)

    chk = sub.add_parser("check", help="compare the current model against a fingerprint")
    chk.add_argument("fingerprint", type=Path)
    common(chk)

    args = parser.parse_args(argv)
    if args.command == "baseline":
        if not args.model:
            parser.error("baseline needs --model")
        try:
            return _baseline(args)
        except ProviderError as error:
            print(f"cannot take a baseline: {error}", file=sys.stderr)
            return 2
    return _check(args)


if __name__ == "__main__":
    raise SystemExit(main())
