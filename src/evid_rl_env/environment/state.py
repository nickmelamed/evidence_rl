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
    last_llm_score: float = 0.0
    selected_evidence_ids: set = field(default_factory=set)
    query_count: int = 0
    max_queries: int = 2
    summary: str = ""

    def is_done(self):
        return self.steps_taken >= self.max_steps