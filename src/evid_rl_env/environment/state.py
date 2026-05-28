from dataclasses import dataclass, field

@dataclass
class Evidence:
    id: int
    text: str
    label: str

@dataclass
class State:
    claim: str
    evidence_pool: list
    selected_evidence: list = field(default_factory=list)
    debate_history: list = field(default_factory=list)
    steps_taken: int = 0
    max_steps: int = 10

    def is_done(self):
        return self.steps_taken >= self.max_steps