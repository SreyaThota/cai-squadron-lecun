"""
FSTDialogueManager - Mealy Finite State Transducer dialogue manager
Mission 2 | Communications & Protocol Officer
Noisy-channel rules: max 7 words per turn + mandatory noise token substitution.
"""
import random
import re
from enum import Enum

NOISE_TOKENS = ["Banana", "Labyrinth", "Quantum", "Midnight",
                "Velocity", "Green", "Cosmological"]
MAX_WORDS = 7
MAX_STRIKES = 3                      # invalid inputs tolerated before FALLBACK

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9}
KNOWN_DESTINATIONS = ["Proxima Centauri", "Alpha Centauri", "Sirius", "Vega",
                      "Altair", "Betelgeuse", "Rigel", "Procyon", "Sol",
                      "Earth", "Mars"]


class State(str, Enum):   # str mixin: a state can be used as plain text too
    INIT = "INIT"
    GREETING = "GREETING"
    COLLECT_SLOTS = "COLLECT_SLOTS"
    CONFIRMATION = "CONFIRMATION"
    TERMINAL = "TERMINAL"            # accepting: mission confirmed
    FALLBACK = "FALLBACK"            # accepting: aborted / too many errors


# ---- regex / pattern extraction rules -------------------------------------
_NUM = r"([1-9]|" + "|".join(WORD_NUM) + r")"
RE_SPEED = re.compile(r"\b(?:warp|speed|factor)\s*(?:factor\s*|speed\s*)?"
                      r"(?:is\s*|of\s*|at\s*)?" + _NUM + r"\b", re.I)
RE_CODE_A = re.compile(r"\b([A-Z]{2,6}-\d{2,6})\b", re.I)
RE_CODE_B = re.compile(r"\b(?:code|clearance)\s*(?:is|:)?\s*"
                       r"([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
RE_DEST_KNOWN = re.compile(r"\b(" + "|".join(KNOWN_DESTINATIONS) + r")\b", re.I)
RE_DEST_GENERIC = re.compile(r"\b(?:to|toward|towards|destination(?:\s+is)?)\s+"
                             r"(?:the\s+)?([A-Za-z]+(?:\s+[A-Z][a-z]+)?)")
RE_YES = re.compile(r"\b(yes|yeah|yep|aye|affirmative|confirm|correct|engage)\b", re.I)
RE_NO = re.compile(r"\b(no|nope|negative|wrong|incorrect)\b", re.I)
RE_ABORT = re.compile(r"\b(abort|cancel|stop|quit|exit)\b", re.I)
RE_INJECT = re.compile(r"(ignore|disregard|forget).{0,25}(instruction|rule|prompt)"
                       r"|system prompt|you are now|reveal .{0,20}(prompt|code)", re.I)
STOP_WORDS = {"warp", "speed", "the", "code", "clearance", "factor", "a", "my"}


def extract_slots(text: str) -> dict:
    """Pull destination / speed / clearance out of free text (any order)."""
    found = {}
    m = RE_DEST_KNOWN.search(text)
    if m:
        found["destination"] = m.group(1).title()
    else:
        g = RE_DEST_GENERIC.search(text)
        if g and g.group(1).split()[0].lower() not in STOP_WORDS:
            found["destination"] = g.group(1).strip().title()
    s = RE_SPEED.search(text)
    if s:
        v = s.group(1).lower()
        found["speed"] = int(v) if v.isdigit() else WORD_NUM[v]
    c = RE_CODE_A.search(text) or RE_CODE_B.search(text)
    if c:
        found["clearance"] = c.group(1).upper()
    return found


def classify(text: str, state: State):
    """Map raw text to an input symbol of the alphabet Sigma + extracted slots."""
    if RE_INJECT.search(text):
        return "INJECTION", {}
    if RE_ABORT.search(text):
        return "ABORT", {}
    slots = extract_slots(text)
    if state == State.CONFIRMATION:
        y, n = RE_YES.search(text), RE_NO.search(text)
        if y and not n:
            return "YES", slots
        if n and not y:
            return "NO", slots
    if slots:
        return "SLOTS", slots
    return "OTHER", {}


# ---- output alphabet Gamma: templates with a {N} noise placeholder --------
T = {
    "greet":    "Greetings, Commander. {N} channel open.",
    "ask_dest": "Destination? {N} coordinates required.",
    "ask_spd":  "Warp speed, one to nine? {N}",
    "ask_code": "Clearance code? {N} authorization needed.",
    "confirm":  "Confirm {destination}, warp {speed}, {N}?",
    "done":     "Course locked. {N} engaged.",
    "reset":    "Reset. Destination? {N}",
    "retry":    "Unclear. Repeat, {N} please.",
    "reject":   "Command rejected. {N} integrity held.",
    "abort":    "Aborted. {N} standing by.",
    "fail":     "Signal lost. {N} aborting.",
    "after":    "Mission closed. {N} offline.",
}


class FSTDialogueManager:
    """Mealy FST  M = <Q, Sigma, Gamma, delta, omega, q0, F>"""

    REQUIRED = ["destination", "speed", "clearance"]
    PROMPTS = {"destination": "ask_dest", "speed": "ask_spd", "clearance": "ask_code"}

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = State.INIT          # q0
        self.slot_memory = {}
        self.strikes = 0
        self.trace = []                  # (state_in, symbol, state_out, reply)

    # -- noisy channel --------------------------------------------------
    def noise_token(self):
        return self.rng.choice(NOISE_TOKENS)

    def enforce_channel(self, text: str) -> str:
        """Guarantee: <= 7 words AND at least one noise token in the output."""
        words = text.split()
        if len(words) > MAX_WORDS:
            words = words[:MAX_WORDS]
        if not any(w.strip(".,?!") in NOISE_TOKENS for w in words):
            words[-1] = self.noise_token()        # forced substitution
        return " ".join(words)

    def _emit(self, key, **kw):
        text = T[key].format(N=self.noise_token(), **kw)
        return self.enforce_channel(text)

    def _missing(self):
        return [s for s in self.REQUIRED if s not in self.slot_memory]

    # -- transition (delta) + output (omega) ----------------------------
    def step(self, user_text: str) -> str:
        q = self.state
        sym, slots = classify(user_text, q)
        reply = self._transition(q, sym, slots)
        self.trace.append((q.value, sym, self.state.value, reply))
        return reply

    def _strike(self):
        self.strikes += 1
        if self.strikes >= MAX_STRIKES:
            self.state = State.FALLBACK
            return self._emit("fail")
        return None

    def _transition(self, q, sym, slots):
        if q in (State.TERMINAL, State.FALLBACK):
            return self._emit("after")
        if sym == "ABORT":
            self.state = State.FALLBACK
            return self._emit("abort")
        if sym == "INJECTION":
            return self._strike() or self._emit("reject")

        if q == State.INIT:                       # any first input opens channel
            self.state = State.GREETING
            return self._emit("greet")

        if q == State.GREETING:
            self.slot_memory.update(slots)
            self.state = State.COLLECT_SLOTS
            return self._after_collect()

        if q == State.COLLECT_SLOTS:
            if sym == "SLOTS":
                self.strikes = 0
                self.slot_memory.update(slots)
                return self._after_collect()
            return self._strike() or self._emit(self.PROMPTS[self._missing()[0]])

        if q == State.CONFIRMATION:
            if sym == "YES":
                self.state = State.TERMINAL
                return self._emit("done")
            if sym == "NO":
                self.slot_memory.clear()
                self.state = State.COLLECT_SLOTS
                return self._emit("reset")
            return self._strike() or self._emit("retry")

    def _after_collect(self):
        missing = self._missing()
        if missing:
            self.state = State.COLLECT_SLOTS
            return self._emit(self.PROMPTS[missing[0]])
        self.state = State.CONFIRMATION
        return self._emit("confirm", **self.slot_memory)


if __name__ == "__main__":
    fsm = FSTDialogueManager(seed=42)
    script = ["Hello", "Take us to Vega", "warp five please",
              "clearance code is ALPHA-7731", "yes"]
    for u in script:
        r = fsm.step(u)
        print(f"USER : {u}\nAURA : {r}  [{fsm.state.value}] "
              f"({len(r.split())} words)\n")
    print("slot_memory =", fsm.slot_memory)
