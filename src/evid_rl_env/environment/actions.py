class Actions:
    SELECT = "select_evidence"
    REMOVE = "remove_evidence"
    SUPPORT = "generate_support_argument"
    CONTRADICT = "generate_contradict_argument"
    FINALIZE = "finalize"

ACTIONS = [
    Actions.SELECT,
    Actions.REMOVE,
    Actions.SUPPORT,
    Actions.CONTRADICT,
    Actions.FINALIZE
]