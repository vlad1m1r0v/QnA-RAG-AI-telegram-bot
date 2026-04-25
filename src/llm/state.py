from typing import TypedDict, Literal
from langgraph.graph import MessagesState


class BriefState(TypedDict):
    project_type: str | None
    project_description: str | None
    goals: list[str]
    key_features: list[str]
    additional_features: list[str]
    integrations: list[str]
    client_materials: list[str]


class RejectedFieldInfo(TypedDict):
    options: list[str]              # short option labels offered for this field
    counts: int                     # how many times user rejected options for this field
    last_rejection_message_id: str  # id of the last human message that triggered a rejection count


class AgentState(MessagesState):
    summary: str | None
    brief: BriefState
    rejected_options: dict[str, RejectedFieldInfo]
    brief_status: Literal["in_progress", "complete"]
    empty_fields: list[str]
    # Inter-node communication
    qna_response: str | None
    has_question: bool
    has_project_info: bool
    is_nonsense: bool
    # Response info
    response_type: Literal["brief_clarifying", "brief_ready", "estimation"]
    estimation: str | None