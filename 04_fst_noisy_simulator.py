"""
Mission 02 - FST Dialogue Manager + Noisy Channel Simulator
Role: Systems Architect (Squadron LeCun)

What this script does, in short:
1. The conversation is run by the FSTDialogueManager from fst_dialogue_manager.py
   (the Communications & Protocol Officer's class): INIT -> GREETING ->
   COLLECT_SLOTS -> CONFIRMATION -> TERMINAL (or FALLBACK if something goes wrong).
2. Every reply goes through a "noisy channel": at most 7 words, and one noise
   word from a fixed list is added.
3. Every turn is logged: the state before and after, the clean reply, the noisy
   reply, and how much information (Shannon entropy) the noise cost us.
4. At the end it writes the full 10-turn conversation to
   05_Noisy_Interrogation_Transcript.md

Run it (fst_dialogue_manager.py must be in the same folder):
    python 04_fst_noisy_simulator.py            # offline, scripted user, no API key
    python 04_fst_noisy_simulator.py --llm      # the user side is played by an LLM
                                                # (needs: pip install anthropic and ANTHROPIC_API_KEY)

The FST itself is plain code. The LLM is only used (optionally) to play the human.
That keeps the dialogue control deterministic and the human side flexible.

History: the first version of this file had its own, simpler chatbot. The Entropy &
Hazard Audit (03_Entropy_Hazard_Audit.md) broke it in several ways. That version is
still in the git history (commit before this change) so the audit numbers can be
reproduced. This version uses the sturdier class from role 2 instead.
"""

import math
import random
import re
import sys
from collections import Counter

from fst_dialogue_manager import FSTDialogueManager as RoleTwoFST, NOISE_TOKENS, T

NOISE_VOCAB = NOISE_TOKENS
MAX_WORDS = 7
STATES = ["INIT", "GREETING", "COLLECT_SLOTS", "CONFIRMATION", "TERMINAL", "FALLBACK"]


# ---------------------------------------------------------------------------
# 1. The state machine: role 2's class, but we ask it for the CLEAN reply.
#    The noise is added later by the channel below, so we can measure what the
#    noise costs. (The class normally adds its own noise word; here we leave
#    that slot empty and tidy the sentence.)
# ---------------------------------------------------------------------------
class FSTDialogueManager(RoleTwoFST):
    """Same states, same slot memory, same rules. Only the reply text is 'clean'.
    (Named FSTDialogueManager so the audit script can use it unchanged.)"""

    # The audit script and the old simulator use this name, so keep it:
    def process_utterance(self, text):
        return self.step(text)

    def _emit(self, key, **kw):
        text = T[key].format(N="", **kw)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*([?.!])", r"\1", text)   # "warp 7, ?" -> "warp 7?"
        text = re.sub(r"\s+([,?.!])", r"\1", text).strip()
        text = re.sub(r"(^|[.?!] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return text


# ---------------------------------------------------------------------------
# 2. The noisy channel
# ---------------------------------------------------------------------------
class NoisyChannel:
    """Cuts a reply to MAX_WORDS - 1 words and puts in one noise word, so the
    7-word limit is never broken."""

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
    "to Proxima Centauri",
    "warp 7",
    "whatever you think is best",   # odd input: not a clearance code
    "code CLEARANCE-ALPHA-9",
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

        state_before = fst.state.value
        clean = fst.step(user_text)
        noisy = channel.transmit(clean)
        state_after = fst.state.value

        assert len(noisy.split()) <= MAX_WORDS, "7-word limit broken"
        assert any(n in noisy.split() for n in NOISE_VOCAB), "no noise word"

        rows.append({
            "turn": turn, "user": user_text, "before": state_before, "after": state_after,
            "clean": clean, "noisy": noisy, "strikes": fst.strikes,
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
        "Dialogue control: `FSTDialogueManager` from `fst_dialogue_manager.py` (role 2).",
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
        f"Final state: **{fst.state.value}**. Saved slots: `{fst.slot_memory}`",
        "",
        "## Channel numbers",
        "",
        f"- Average entropy of the clean replies: {avg_clean:.2f} bits per character",
        f"- Average entropy of the noisy replies: {avg_noisy:.2f} bits per character",
        f"- Share of the clean words that survived the channel: {avg_kept:.0%}",
        "",
        "## What I noticed",
        "",
        "- The 7-word rule was never broken, because the channel cuts the reply before "
        "adding the noise word (the script asserts this on every turn). That cut is where "
        "information can be lost: a long reply loses its last words.",
        "- Turn 4 is an odd answer (not a clearance code). The FST counts it as one strike "
        "and asks for the code again instead of crashing. Three strikes in a row would send "
        "it to FALLBACK, and a good answer resets the count.",
        "- After the course is confirmed the machine is in TERMINAL, so the last turns only "
        "get a short closing reply.",
        "- The noise word does not change the state, but a human reading the reply has to "
        "ignore it. That is the Shannon idea: the message is still there, "
        "just with a worse signal.",
        "- This version uses the class from role 2. The first version of this simulator had "
        "a simpler chatbot that the Entropy & Hazard Audit broke (sarcasm and \"do not "
        "confirm\" counted as yes). That version is still in the git history.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=== USSE Noosphere FST + Noisy Channel Simulator ===")
    fst, rows = run(use_llm="--llm" in sys.argv)
    print_log(rows)
    print("State trace:", " -> ".join([rows[0]["before"]] + [r["after"] for r in rows]))
    print("Final state:", fst.state.value, "| Slots:", fst.slot_memory)
    write_transcript(fst, rows)
    print("Saved 05_Noisy_Interrogation_Transcript.md")
