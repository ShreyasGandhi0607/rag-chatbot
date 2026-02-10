from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class Task(BaseModel):
    topic: str
    intent: str
    domain_name: Optional[str] = None


class GlobalState(BaseModel):
    messages: List[Dict]

    tasks: List[Task] = Field(default_factory=list)
    current_task_index: int = 0

    topic: Optional[str] = None
    intent: Optional[str] = None
    domain_name: Optional[str] = None

    account_id: Optional[str] = None
    auth_code: Optional[str] = None
    settings_type: Optional[str] = None

    missing_fields: List[str] = Field(default_factory=list)