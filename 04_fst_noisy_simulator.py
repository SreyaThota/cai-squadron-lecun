"""
Mission 02 - FST Dialogue Manager + Noisy Channel Simulator
Role: Systems Architect (Squadron LeCun)

What this script does, in short:
1. A small state machine (FST) runs the conversation: INIT -> GREETING ->
   COLLECT_SLOTS -> CONFIRMATION -> TERMINAL (or FALLBACK if something goes wrong).
2. Every reply goes through a "noisy channel": at most 7 words, and one noise
   word from a fixed list is added.
3. Every turn is logged: the state before and after, the clean reply, the noisy
   reply, and how much information (Shannon entropy) the noise cost us.
4. At the end it writes the full 10-turn conversation to
   05_Noisy_Interrogation_Transcript.md

Run it:
    python 04_fst_noisy_simulator.py            # offline, scripted user, no API key
    python 04_fst_noisy_simulator.py --llm      # the user side is played by an LLM
                                                # (needs: pip install anthropic and ANTHROPIC_API_KEY)

The FST itself is plain code. The LLM is only used (optionally) to play the human.
That keeps the dialogue control deterministic and the human side flexible.
"""

import math
import random
import re
import sys
from collections import Counter

NOISE_VOCAB = ["Banana", "Labyrinth", "Quantum", "Midnight", "Velocity", "Green", "Cosmological"]
MAX_WORDS = 7
STATES = ["INIT", "GREETING", "COLLECT_SLOTS", "CONFIRMATION", "TERMINAL", "FALLBACK"]


# ---------------------------------------------------------------------------
# 1. The state machine
# ---------------------------------------------------------------------------
class FSTDialogueManager:
    """Deterministic dialogue control. Same input in the same state -> same output."""

    def __init__(self):
        self.state = "INIT"
        self.slot_memory = {}
        self.fallback_count = 0

    def process_utterance(self, user_input: str) -> str:
        text = user_input.strip()
        upper = text.upper()

        if self.state == "INIT":
            self.state = "GREETING"
            return "AURA online. Where are we going?"

        if self.state == "GREETING":
            destination = self._extract_destination(text)
            if not destination:
                return self._bad_input("Destination unclear. Repeat it.")
            self.slot_memory["destination"] = destination
            self.fallback_count = 0
            self.state = "COLLECT_SLOTS"
            return f"Destination {destination}. Which warp speed?"

        if self.state == "COLLECT_SLOTS":
            if "speed" not in self.slot_memory:
                match = re.search(r"\b([1-9])\b", text)
                if not match:
                    return self._bad_input("Speed must be warp one to nine.")
                self.slot_memory["speed"] = int(match.group(1))
                self.fallback_count = 0
                return "Speed saved. Clearance code, please?"
            match = re.search(r"[A-Z]+-[A-Z]+-\d+", upper)
            if not match:
                return self._bad_input("Clearance code format: WORD-WORD-NUMBER.")
            self.slot_memory["clearance"] = match.group(0)
            self.fallback_count = 0
            self.state = "CONFIRMATION"
            return (f"Confirm {self.slot_memory['destination']}, "
                    f"warp {self.slot_memory['speed']}? Yes or no?")

        if self.state == "CONFIRMATION":
            if re.search(r"\b(YES|YEAH|CONFIRM|OK)\b", upper):
                self.state = "TERMINAL"
                return "Course locked. Mission complete."
            self.state = "FALLBACK"
            return "Not confirmed. Falling back."

        if self.state == "FALLBACK":
            return "Fallback active. Restart needed."
        return "Done. Channel closed."

    def _bad_input(self, message: str) -> str:
        """Count bad inputs. After 2 in a row we go to FALLBACK."""
        self.fallback_count += 1
        if self.fallback_count >= 2:
            self.state = "FALLBACK"
            return "Too many errors. Falling back."
        return message

    @staticmethod
    def _extract_destination(text: str) -> str:
        # Keep it simple: a destination is 1-3 words made of letters/digits.
        cleaned = re.sub(r"(?i)^(go to|take us to|to|we go to|destination is)\s+", "", text.strip())
        cleaned = re.sub(r"[^A-Za-z0-9 ]", "", cleaned).strip()
        if 0 < len(cleaned.split()) <= 3:
            return cleaned.title()
        return ""


# ---------------------------------------------------------------------------
# 2. The noisy channel
# ---------------------------------------------------------------------------
class NoisyChannel:
    """Cuts a reply to MAX_WORDS and puts in one noise word (it first cuts the reply to 6 words, so the 7-word limit is never broken)."""

    def __init__(self, seed=7):
        self.rng = random.Random(seed)  # fixed seed -> same transcript every run

    def transmit(self, clean_reply: str) -> str:
        words = clean_reply.split()[: MAX_WORDS - 1]
        noise = self.rng.choice(NOISE_VOCAB)
        position = self.rng.randint(0, len(words))
        words.insert(position, noise)
        return " ".join(words)


# ---------------------------------------------------------------------------
# 3. Entropy helpers (Shannon)
# ---------------------------------------------------------------------------
def entropy(text: str) -> float:
    """Shannon entropy of the characters in text, in bits per character."""
    chars = [c for c in text.lower() if not c.isspace()]
    if not chars:
        return 0.0
    counts = Counter(chars)
    total = len(chars)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def kept_info(clean: str, noisy: str) -> float:
    """Share of the clean reply's words that survived in the noisy reply."""
    clean_words = [w.strip(".,?!").lower() for w in clean.split()]
    noisy_words = {w.strip(".,?!").lower() for w in noisy.split()}
    if not clean_words:
        return 1.0
    return sum(w in noisy_words for w in clean_words) / len(clean_words)


# ---------------------------------------------------------------------------
# 4. The human side (scripted by default, LLM if --llm)
# ---------------------------------------------------------------------------
# 10 user turns. Turn 4 is a deliberately odd input to test the FST.
SCRIPTED_USER = [
    "Hello",
    "Proxima Centauri",
    "Warp 7",
    "whatever you think is best",   # odd input: not a clearance code
    "CLEARANCE-ALPHA-9",
    "yes",
    "thanks",
    "anything else?",
    "bye",
    "hello again?",
]


def llm_user_turn(history):
    """Optional: let an LLM play the human. Falls back to the script if it fails."""
    import anthropic  # imported here so offline mode needs no packages

    client = anthropic.Anthropic()
    prompt = ("You are a ship cadet talking to a very short-spoken computer. "
              "Answer its last message in under 8 words. Conversation so far:\n" + "\n".join(history))
    reply = client.messages.create(model="claude-sonnet-5-5", max_tokens=40,
                                   messages=[{"role": "user", "content": prompt}])
    return reply.content[0].text.strip()


# ---------------------------------------------------------------------------
# 5. The simulator
# ---------------------------------------------------------------------------
def run(use_llm=False):
    fst = FSTDialogueManager()
    channel = NoisyChannel()
    rows = []
    history = []

    for turn, scripted in enumerate(SCRIPTED_USER, start=1):
        if use_llm and turn > 1 and history:
            try:
                user_text = llm_user_turn(history)
            except Exception as error:  # no key, no network, etc.
                print(f"[LLM unavailable ({error}); using script]")
                user_text = scripted
        else:
            user_text = scripted

        state_before = fst.state
        clean = fst.process_utterance(user_text)
        noisy = channel.transmit(clean)
        state_after = fst.state

        assert len(noisy.split()) <= MAX_WORDS, "7-word limit broken"
        assert any(n in noisy.split() for n in NOISE_VOCAB), "no noise word"

        rows.append({
            "turn": turn, "user": user_text, "before": state_before, "after": state_after,
            "clean": clean, "noisy": noisy,
            "h_clean": entropy(clean), "h_noisy": entropy(noisy),
            "kept": kept_info(clean, noisy),
        })
        history += [f"User: {user_text}", f"Computer: {noisy}"]

    return fst, rows


def print_log(rows):
    for r in rows:
        print(f"Turn {r['turn']:>2} | {r['before']:<13} -> {r['after']:<13} | "
              f"user: {r['user']!r} | sent: {r['noisy']!r} | "
              f"H {r['h_clean']:.2f}->{r['h_noisy']:.2f} bits/char | kept {r['kept']:.0%}")


def write_transcript(fst, rows, path="05_Noisy_Interrogation_Transcript.md"):
    lines = [
        "# Mission 2 - Noisy Interrogation Transcript",
        "",
        "Squad: Squadron LeCun. Role: Systems Architect.",
        "Rules: replies are at most 7 words and carry one noise word "
        "(" + ", ".join(NOISE_VOCAB) + ").",
        "",
        "| Turn | User says | State change | Clean reply | Sent over noisy channel | Kept |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['turn']} | {r['user']} | {r['before']} -> {r['after']} | "
                     f"{r['clean']} | {r['noisy']} | {r['kept']:.0%} |")

    avg_clean = sum(r["h_clean"] for r in rows) / len(rows)
    avg_noisy = sum(r["h_noisy"] for r in rows) / len(rows)
    avg_kept = sum(r["kept"] for r in rows) / len(rows)
    trace = " -> ".join([rows[0]["before"]] + [r["after"] for r in rows])

    lines += [
        "",
        "## State trace",
        "",
        trace,
        "",
        f"Final state: **{fst.state}**. Saved slots: `{fst.slot_memory}`",
        "",
        "## Channel numbers",
        "",
        f"- Average entropy of the clean replies: {avg_clean:.2f} bits per character",
        f"- Average entropy of the noisy replies: {avg_noisy:.2f} bits per character",
        f"- Share of the clean words that survived the channel: {avg_kept:.0%}",
        "",
        "## What I noticed",
        "",
        "- The state machine never broke the 7-word rule, because the channel cuts the "
        "reply before adding the noise word. That cut is where information is lost: "
        "long replies lose their last words.",
        "- Turn 4 is an odd answer (not a clearance code). The FST counts it as a bad "
        "input and asks again instead of crashing. Two bad inputs in a row would send it to FALLBACK.",
        "- After the course is confirmed the machine is in TERMINAL, so the last turns only "
        "get a short closing reply.",
        "- The noise word does not change the state, but a human reading the reply has to "
        "ignore it. That is the Shannon idea: the message is still there, "
        "just with a worse signal.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=== USSE Noosphere FST + Noisy Channel Simulator ===")
    fst, rows = run(use_llm="--llm" in sys.argv)
    print_log(rows)
    print("State trace:", " -> ".join([rows[0]["before"]] + [r["after"] for r in rows]))
    print("Final state:", fst.state, "| Slots:", fst.slot_memory)
    write_transcript(fst, rows)
    print("Saved 05_Noisy_Interrogation_Transcript.md")
