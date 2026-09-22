"""Live smoke test against a real backend (OpenRouter or Ollama).

Complements the unit tests with a run against a real API, including a
needle-in-a-haystack test that proves the compression carries content rather
than merely counting tokens.

    python main.py                                    # proxy in terminal 1
    python scripts/live_check.py openrouter/<model>   # check in terminal 2

The proxy must be running; the API key is read exclusively from its .env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

Json = dict[str, object]

NEEDLE = "ZEBRA-7719"
METRIC_HEADERS = (
    "x-original-tokens",
    "x-compressed-tokens",
    "x-saved-tokens",
    "x-saved-ratio",
    "x-compression-strategy",
    "x-messages-in",
    "x-messages-out",
    "x-provider",
    "x-model",
    "x-route-source",
    "x-tokenizer",
    "x-compression-ms",
    "x-upstream-ms",
)


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_metrics(response: httpx.Response) -> None:
    for key in METRIC_HEADERS:
        if key in response.headers:
            print(f"  {key:26} {response.headers[key]}")


def content_of(payload: Json) -> str:
    choices = payload["choices"]
    assert isinstance(choices, list)
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return str(content).strip()


def post(
    client: httpx.Client,
    base: str,
    body: Json,
    headers: dict[str, str] | None = None,
    attempts: int = 4,
) -> httpx.Response:
    response = client.post(f"{base}/v1/chat/completions", json=body, headers=headers or {})
    for attempt in range(1, attempts):
        if response.status_code != 429:
            return response
        upstream = (
            (response.json().get("error", {}).get("details", {}) or {}).get("upstream_body", {})
            or {}
        ).get("error", {})
        wait = min(15, int((upstream.get("metadata") or {}).get("retry_after_seconds") or 5))
        print(f"  … 429 - waiting {wait}s (attempt {attempt}/{attempts - 1})")
        time.sleep(wait)
        response = client.post(f"{base}/v1/chat/completions", json=body, headers=headers or {})
    return response


def haystack(turns: int) -> list[Json]:
    topics = [
        "the staging deployments",
        "the database migration",
        "the CI pipeline",
        "the rollback plan",
        "the monitoring alerts",
        "secret management",
    ]
    fact = (
        f"Important, please note this down: our emergency rollback code is "
        f"{NEEDLE}. The owner is Mira Kowalski, extension 4412."
    )
    confirmation = f"Noted: rollback code {NEEDLE}, owner Mira Kowalski (extension 4412)."
    messages: list[Json] = [
        {"role": "system", "content": "You are a DevOps assistant. Answer concisely."},
        {"role": "user", "content": fact},
        {"role": "assistant", "content": confirmation},
    ]
    for index in range(turns):
        topic = topics[index % len(topics)]
        question = f"Round {index}: how did we implement {topic}? Include commands. " * 3
        answer = f"Round {index}: we rolled {topic} out behind a feature flag. " * 3
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append(
        {
            "role": "user",
            "content": "Final question: what is our emergency rollback code and who owns it?",
        }
    )
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="e.g. openrouter/google/gemma-4-31b-it:free")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--turns", type=int, default=45, help="length of the haystack")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    client = httpx.Client(timeout=args.timeout)
    failures: list[str] = []

    head("1) /healthz")
    health = client.get(f"{args.base}/healthz").json()
    print(json.dumps(health, indent=2))

    head("2) /v1/models")
    models = client.get(f"{args.base}/v1/models").json()["data"]
    free = sum(m["id"].endswith(":free") for m in models)
    print(f"  Catalogue: {len(models)} models, {free} of them free")

    head("3) Short request (no compression expected)")
    short = post(
        client,
        args.base,
        {
            "model": args.model,
            "max_tokens": 80,
            "messages": [
                {"role": "user", "content": "Answer in one sentence: what is a reverse proxy?"}
            ],
        },
    )
    print(f"  HTTP {short.status_code}")
    show_metrics(short)
    if short.status_code == 200:
        print(f"  Answer: {content_of(short.json())[:200]}")
        if short.headers.get("x-compression-strategy") != "none":
            failures.append("short request was compressed unexpectedly")
    else:
        failures.append(f"short request: HTTP {short.status_code}")

    head("4) Streaming (SSE)")
    with client.stream(
        "POST",
        f"{args.base}/v1/chat/completions",
        json={
            "model": args.model,
            "stream": True,
            "max_tokens": 80,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        },
    ) as response:
        print(f"  HTTP {response.status_code} | {response.headers.get('content-type')}")
        show_metrics(response)
        text = "".join(response.iter_text())
    frames = [line for line in text.splitlines() if line.startswith("data: ")]
    print(f"  SSE frames: {len(frames)} | last: {frames[-1] if frames else '-'}")
    if not frames or frames[-1] != "data: [DONE]":
        failures.append("stream did not end with [DONE]")
    else:
        streamed = "".join(
            json.loads(f[6:])["choices"][0]["delta"].get("content") or ""
            for f in frames
            if f != "data: [DONE]"
        )
        print(f"  reassembled: {streamed.strip()[:160]}")

    head("5) Needle in a haystack - does compression carry the content?")
    messages = haystack(args.turns)
    print(f"  Messages: {len(messages)} | needle '{NEEDLE}' sits in message 2")
    started = time.perf_counter()
    long_response = post(
        client,
        args.base,
        {"model": args.model, "max_tokens": 200, "messages": messages},
        headers={"X-Session-Id": "live-check"},
    )
    print(f"  HTTP {long_response.status_code} after {time.perf_counter() - started:.1f}s")
    show_metrics(long_response)
    if long_response.status_code != 200:
        failures.append(f"long request: HTTP {long_response.status_code}")
        print(long_response.text[:400])
    else:
        answer = content_of(long_response.json())
        print(f"\n  ANSWER: {answer[:400]}")
        if float(long_response.headers.get("x-saved-ratio", 0)) <= 0:
            failures.append("no tokens were saved")
        if NEEDLE in answer:
            print(f"\n  ⇒ '{NEEDLE}' survived the compression.")
        else:
            failures.append(f"needle '{NEEDLE}' was lost during compression")

    head("6) /v1/stats")
    stats = client.get(f"{args.base}/v1/stats").json()
    for key in (
        "total_requests",
        "successful_requests",
        "failed_requests",
        "compressed_requests",
        "total_saved_tokens",
        "overall_saved_ratio",
        "avg_total_latency_ms",
    ):
        print(f"  {key:26} {stats[key]}")

    head("RESULT")
    if failures:
        for failure in failures:
            print(f"  FAILED: {failure}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
