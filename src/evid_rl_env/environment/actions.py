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
]