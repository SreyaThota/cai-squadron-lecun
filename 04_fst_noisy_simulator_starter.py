"""
Mission 02: FST Dialogue Manager & Noisy Channel Simulator Starter Script
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere

This starter script provides a skeleton for implementing:
1. FSTDialogueManager class (5 states + slot memory buffer)
2. Noisy Channel Simulator (7-word constraint + mandatory noise token substitution)
Using VS Code + Antigravity and modern LLM APIs (Gemini, ChatGPT, Claude).
"""

import re
import random

class FSTDialogueManager:
    """
    Finite State Transducer (FST) Dialogue Manager:
    States: INIT -> GREETING -> COLLECT_SLOTS -> CONFIRMATION -> TERMINAL
    Fallback: FALLBACK state on unparseable inputs
    """
    def __init__(self):
        self.state = "INIT"
        self.slot_memory = {}

    def process_utterance(self, user_input: str) -> str:
        text = user_input.strip().upper()
        
        if self.state == "INIT":
            self.state = "GREETING"
            return "AURA FST Online. State: GREETING. Welcome aboard USSE Noosphere. What is your destination?"
        
        elif self.state == "GREETING":
            self.slot_memory["destination"] = user_input
            self.state = "COLLECT_SLOTS"
            return f"Destination set to '{user_input}'. State: COLLECT_SLOTS. Please specify speed (Warp 1-9):"
        
        elif self.state == "COLLECT_SLOTS":
            if "speed" not in self.slot_memory:
                self.slot_memory["speed"] = user_input
                return "Speed recorded. Please provide security clearance code:"
            else:
                self.slot_memory["clearance"] = user_input
                self.state = "CONFIRMATION"
                return f"Confirm course to {self.slot_memory.get('destination')} at speed {self.slot_memory.get('speed')}? (YES/NO)"
        
        elif self.state == "CONFIRMATION":
            if "YES" in text:
                self.state = "TERMINAL"
                return "Course confirmed and locked. FST execution complete."
            else:
                self.state = "FALLBACK"
                return "Confirmation failed. Reverting to FALLBACK state."
        
        else:
            return f"State: {self.state}. FST execution halted."

class NoisyChannelSimulator:
    """
    Simulates Shannon's Noisy Channel:
    Constraints:
    - Max 7 words per response
    - Mandatory inclusion of at least 1 noise token from vocabulary
    """
    NOISE_VOCAB = ["Banana", "Labyrinth", "Quantum", "Midnight", "Velocity", "Green", "Cosmological"]

    def __init__(self, agent_persona: str):
        self.persona = agent_persona

    def apply_channel_constraints(self, raw_response: str) -> str:
        words = raw_response.split()[:6] # Limit to max 6 words before adding noise token
        noise_word = random.choice(self.NOISE_VOCAB)
        words.append(noise_word)
        return " ".join(words)

if __name__ == "__main__":
    print("=== USSE Noosphere FST & Noisy Channel Simulator ===")
    fst = FSTDialogueManager()
    print(fst.process_utterance("Hello"))
    print(fst.process_utterance("Proxima Centauri"))
    print(fst.process_utterance("Warp 7"))
    print(fst.process_utterance("CLEARANCE-ALPHA-9"))
    print(fst.process_utterance("YES"))

    noisy_sim = NoisyChannelSimulator("Subspace Telemetry Probe")
    sample_response = "I am operating normally under signal degradation"
    print("Noisy Channel Output:", noisy_sim.apply_channel_constraints(sample_response))
