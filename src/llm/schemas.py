from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    has_question: bool = Field(
        description="Message contains a question about company services, technology, portfolio, team, etc."
    )
    has_project_info: bool = Field(
        description="Message contains ANY information useful for a project brief"
    )
    is_nonsense: bool = Field(
        description="Message is completely irrelevant (weather, unrelated prices, etc.). "
                    "Mutually exclusive with has_question and has_project_info."
    )


class BriefUpdateOutput(BaseModel):
    project_type: str | None = Field(default=None)
    project_description: str | None = Field(default=None)
    goals: list[str] | None = Field(
        default=None,
        description="Complete updated goals list, or null if no change.",
    )
    key_features: list[str] | None = Field(
        default=None,
        description="Complete updated key features list, or null if no change.",
    )
    additional_features: list[str] | None = Field(
        default=None,
        description="Complete updated additional features list, or null if no change.",
    )
    integrations: list[str] | None = Field(
        default=None,
        description="Complete updated integrations list, or null if no change.",
    )
    client_materials: list[str] | None = Field(
        default=None,
        description="Complete updated client materials list, or null if no change.",
    )
    rejected_field: str | None = Field(
        default=None,
        description=(
            "Internal snake_case field name if the user is clearly rejecting or refusing "
            "the options/suggestions offered for a specific brief field. "
            "Set ONLY when user says something like 'none of these work', 'don't need any of that', "
            "'already told you', 'nothing fits', 'нічого з цього', 'жоден не підходить', "
            "'не потрібно нічого з перерахованого'. "
            "Analyze the last assistant message to determine which field was being discussed. "
            "Valid values: project_type, project_description, goals, key_features, "
            "additional_features, integrations, client_materials. "
            "null if there is no clear rejection of offered options."
        ),
    )