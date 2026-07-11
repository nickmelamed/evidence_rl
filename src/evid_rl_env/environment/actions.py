class Actions:
    SELECT = "select_evidence"
    REMOVE = "remove_evidence"
    SUPPORT = "generate_support_argument"
    CONTRADICT = "generate_contradict_argument"
    FINALIZE = "finalize"
    QUERY = "query_evidence"
    RERANK = "rerank_evidence"
    SUMMARIZE = "summarize_evidence"
    CONCEDE = "concede_point"
    ASSIGN_CONFIDENCE = "assign_confidence"
    CHALLENGE_EVIDENCE = "challenge_evidence"
    REQUEST_CLARIFICATION = "request_clarification"
    HEDGE = "hedge"

ACTIONS = [
    Actions.SELECT,
    Actions.REMOVE,
    Actions.SUPPORT,
    Actions.CONTRADICT,
    Actions.FINALIZE,
    Actions.QUERY,
    Actions.RERANK,
    Actions.SUMMARIZE,
    Actions.CONCEDE,
    Actions.ASSIGN_CONFIDENCE,
    Actions.CHALLENGE_EVIDENCE,
    Actions.REQUEST_CLARIFICATION,
    Actions.HEDGE,
]