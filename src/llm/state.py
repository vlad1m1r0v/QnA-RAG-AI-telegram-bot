from typing import Literal
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    summary: str
    project_type: str | None
    project_description: str | None
    goals: list[str]
    key_features: list[str]
    additional_features: list[str]
    integrations: list[str]
    client_materials: list[str]
    estimation: str | None
    brief_status: Literal["in_progress", "complete"]
    response_type: Literal["brief_clarifying", "brief_ready", "estimation"]
    # Inter-node communication (reset each turn by router_node)
    qna_response: str | None
    empty_fields: list[str]
    weak_fields: list[str]
    # Router flags (set by router_node, read by conditional edges)
    has_question: bool
    has_project_info: bool
    is_nonsense: bool