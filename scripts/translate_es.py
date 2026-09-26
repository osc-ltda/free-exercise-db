#!/usr/bin/env python3
"""Fill `nameEs` and `instructionsEs` in exercises/*.json with Gemini.

Resumable: an exercise that already has both fields (with one Spanish step
per English step) is skipped, and every file is written as soon as its
translation comes back, so the script can be stopped at any point and simply
run again - tomorrow, or after the next batch of exercises is added.

Sized for the Gemini free tier: at most --rpm requests a minute and --rpd a
day. The daily count is kept in scripts/.translate_es_state.json, since the
quota outlives the process; it resets at midnight Pacific, which is when
Google resets it. Several exercises go in one request (--batch) so the whole
catalogue fits in far fewer requests than it has files.

Usage:
    export GEMINI_API_KEY=...
    python3 scripts/translate_es.py              # translate until done or out of quota
    python3 scripts/translate_es.py --dry-run    # just count what is left
    python3 scripts/translate_es.py --only Barbell_Squat --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
EXERCISES_DIR = ROOT / "exercises"
STATE_FILE = Path(__file__).resolve().parent / ".translate_es_state.json"
QUOTA_TZ = ZoneInfo("America/Los_Angeles")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Pinned so the same movement reads the same way across 888 files (issue #30).
GLOSSARY = {
    "Barbell": "Barra",
    "Dumbbell": "Mancuerna",
    "Kettlebell": "Pesa rusa",
    "Cable": "Polea",
    "Machine": "Máquina",
    "Smith Machine": "Máquina Smith",
    "Bench Press": "Press de banca",
    "Deadlift": "Peso muerto",
    "Squat": "Sentadilla",
    "Row": "Remo",
    "Curl": "Curl",
    "Raise": "Elevación",
    "Extension": "Extensión",
    "Pulldown": "Jalón",
    "Press": "Press",
    "Fly": "Apertura",
    "Lunge": "Zancada",
    "Calf Raise": "Elevación de pantorrilla",
    "Bodyweight": "Peso corporal",
    "Assisted": "Asistido",
    "Push-Up": "Flexión de brazos",
    "Pull-Up": "Dominada",
    "Chin-Up": "Dominada supina",
    "Dip": "Fondo",
    "Crunch": "Abdominal",
    "Plank": "Plancha",
    "Shrug": "Encogimiento",
    "Stretch": "Estiramiento",
    "Grip": "Agarre",
    "Incline": "Inclinado",
    "Decline": "Declinado",
    "Band": "Banda",
    "E-Z Curl Bar": "Barra Z",
}

PROMPT = """You translate a gym app's exercise catalogue from English to Spanish.

Rules:
- Neutral Latin American Spanish, addressing the reader as "tú".
- Use this glossary for these terms wherever they appear, in names and in
  instructions alike:
{glossary}
- Names are short labels shown in a list: keep them short, keep the
  parenthesised qualifiers in parentheses, e.g.
  "Bench Press (Barbell - Medium Grip)" -> "Press de banca (Barra - Agarre medio)".
  Terms Spanish-speaking lifters use in English (Curl, Press, Hip Thrust,
  Good Morning, Kettlebell Swing...) may stay in English.
- Translate each instruction step on its own: `instructionsEs` must have
  exactly as many items as `instructions`, in the same order. Never merge or
  split steps.
- Keep numbers, units and symbols as they are.
- Return one entry per input exercise, with the same `id`.

Exercises:
{payload}
"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "nameEs": {"type": "STRING"},
            "instructionsEs": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["id", "nameEs", "instructionsEs"],
    },
}


class DailyQuotaExhausted(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    parser.add_argument("--rpm", type=int, default=15, help="requests per minute (default 15)")
    parser.add_argument("--rpd", type=int, default=500, help="requests per day (default 500)")
    parser.add_argument("--batch", type=int, default=5, help="exercises per request (default 5)")
    parser.add_argument("--limit", type=int, help="stop after this many requests in this run")
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these exercise ids")
    parser.add_argument("--force", action="store_true", help="retranslate even if already translated")
    parser.add_argument("--dry-run", action="store_true", help="count what is left and exit")
    return parser.parse_args()


# --- exercises ------------------------------------------------------------


def is_translated(exercise: dict) -> bool:
    es = exercise.get("instructionsEs")
    return bool(exercise.get("nameEs")) and isinstance(es, list) and len(es) == len(exercise["instructions"])


def load_pending(only: list[str] | None, force: bool) -> list[tuple[Path, dict]]:
    pending = []
    for path in sorted(EXERCISES_DIR.glob("*.json")):
        exercise = json.loads(path.read_text(encoding="utf-8"))
        if only and exercise["id"] not in only:
            continue
        if force or not is_translated(exercise):
            pending.append((path, exercise))
    return pending


def write_translation(path: Path, exercise: dict, name_es: str, instructions_es: list[str]) -> None:
    # Rebuild the dict so each Spanish twin sits right after its English field;
    # the files have no trailing newline, and keeping that keeps the diff clean.
    out = {}
    for key, value in exercise.items():
        if key in ("nameEs", "instructionsEs"):
            continue
        out[key] = value
        if key == "name":
            out["nameEs"] = name_es
        elif key == "instructions":
            out["instructionsEs"] = instructions_es
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


# --- quota ----------------------------------------------------------------


class Quota:
    """--rpm as a sliding window in memory, --rpd as a counter on disk."""

    def __init__(self, rpm: int, rpd: int):
        self.rpm = rpm
        self.rpd = rpd
        self.recent: deque[float] = deque()
        self.state = self._load()

    @staticmethod
    def _today() -> str:
        return datetime.now(QUOTA_TZ).date().isoformat()

    def _load(self) -> dict:
        try:
            state = json.loads(STATE_FILE.read_text())
        except (OSError, ValueError):
            state = {}
        if state.get("date") != self._today():
            state = {"date": self._today(), "requests": 0}
        return state

    def _save(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, indent=2) + "\n")

    @property
    def used_today(self) -> int:
        if self.state["date"] != self._today():
            self.state = {"date": self._today(), "requests": 0}
        return self.state["requests"]

    def left_today(self) -> int:
        return max(0, self.rpd - self.used_today)

    def mark_exhausted(self) -> None:
        # Google said the day is over even if our count disagrees (another
        # script on the same key, or a lower real limit); believe Google.
        self.state["requests"] = max(self.used_today, self.rpd)
        self._save()

    def acquire(self) -> None:
        if self.left_today() == 0:
            raise DailyQuotaExhausted
        while True:
            now = time.monotonic()
            while self.recent and now - self.recent[0] >= 60:
                self.recent.popleft()
            if len(self.recent) < self.rpm:
                break
            time.sleep(60 - (now - self.recent[0]) + 0.1)
        self.recent.append(time.monotonic())
        self.state["requests"] = self.used_today + 1
        self._save()


# --- gemini ---------------------------------------------------------------


def text_steps(exercise: dict) -> list[str]:
    # A few source files carry a blank "" step. The model drops or pads those
    # unpredictably, so it only sees the steps with text and the blanks are
    # put back in place afterwards.
    return [step for step in exercise["instructions"] if step.strip()]


def with_blanks(exercise: dict, translated: list[str]) -> list[str]:
    it = iter(translated)
    return [next(it) if step.strip() else step for step in exercise["instructions"]]


def build_prompt(batch: list[dict]) -> str:
    glossary = "\n".join(f"    {en} -> {es}" for en, es in GLOSSARY.items())
    payload = json.dumps(
        [{"id": e["id"], "name": e["name"], "instructions": text_steps(e)} for e in batch],
        indent=2,
        ensure_ascii=False,
    )
    return PROMPT.format(glossary=glossary, payload=payload)


def call_gemini(api_key: str, model: str, prompt: str, quota: Quota) -> list[dict]:
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }
    ).encode()

    for attempt in range(5):
        quota.acquire()
        request = urllib.request.Request(
            API_URL.format(model=model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read())
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return json.loads(text)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code == 429:
                if "PerDay" in detail or "per day" in detail.lower():
                    quota.mark_exhausted()
                    raise DailyQuotaExhausted from None
                wait = 60
            elif error.code >= 500:
                wait = 10 * (attempt + 1)
            else:
                sys.exit(f"Gemini returned {error.code}:\n{detail}")
            print(f"  HTTP {error.code}, retrying in {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as error:
            print(f"  {type(error).__name__}: {error}, retrying in 10s", flush=True)
            time.sleep(10)
    raise RuntimeError("Gemini kept failing, giving up on this batch")


# --- main -----------------------------------------------------------------


def main() -> None:
    args = parse_args()
    pending = load_pending(args.only, args.force)
    quota = Quota(args.rpm, args.rpd)
    requests_needed = -(-len(pending) // args.batch)

    print(
        f"{len(pending)} exercises to translate (~{requests_needed} requests at {args.batch} per request). "
        f"Quota today: {quota.used_today}/{args.rpd} used."
    )
    if args.dry_run or not pending:
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first.")

    queue = deque(pending)
    retry: deque[tuple[Path, dict]] = deque()
    done = failed = requests_sent = 0
    try:
        while queue or retry:
            if args.limit is not None and requests_sent >= args.limit:
                print(f"Reached --limit {args.limit}.")
                break
            if retry:
                batch = [retry.popleft()]
            else:
                batch = [queue.popleft() for _ in range(min(args.batch, len(queue)))]
            by_id = {exercise["id"]: (path, exercise) for path, exercise in batch}
            print(f"[{quota.used_today + 1}/{args.rpd}] {', '.join(by_id)}", flush=True)

            requests_sent += 1
            try:
                results = call_gemini(api_key, args.model, build_prompt([e for _, e in batch]), quota)
            except RuntimeError as error:
                print(f"  {error}")
                failed += len(batch)
                continue

            returned = set()
            for item in results if isinstance(results, list) else []:
                entry = by_id.get(item.get("id"))
                if not entry or item["id"] in returned:
                    continue
                path, exercise = entry
                name_es = (item.get("nameEs") or "").strip()
                steps = [s.strip() for s in item.get("instructionsEs") or []]
                if not name_es or len(steps) != len(text_steps(exercise)) or not all(steps):
                    continue
                steps = with_blanks(exercise, steps)
                write_translation(path, exercise, name_es, steps)
                returned.add(item["id"])
                done += 1
                print(f"  {exercise['name']} -> {name_es}")

            # Anything missing or malformed goes back alone, where the model
            # has less to keep track of; a second miss is left for the next run.
            for exercise_id, entry in by_id.items():
                if exercise_id in returned:
                    continue
                if len(batch) > 1:
                    print(f"  {exercise_id}: bad or missing, retrying on its own")
                    retry.append(entry)
                else:
                    print(f"  {exercise_id}: still bad, skipping for now")
                    failed += 1
    except DailyQuotaExhausted:
        print("\nDaily quota used up. Run the script again tomorrow to carry on.")
    except KeyboardInterrupt:
        print("\nStopped. Everything written so far is kept.")

    left = len(load_pending(args.only, False))
    print(f"\nTranslated {done} this run, {failed} skipped, {left} still left in the catalogue.")


if __name__ == "__main__":
    main()
