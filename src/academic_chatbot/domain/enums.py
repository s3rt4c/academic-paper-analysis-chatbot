from enum import StrEnum


class JobState(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    WAITING_USER = "waiting_user"
    RETRY_WAIT = "retry_wait"
    QUEUED_FOR_RECOVERY = "queued_for_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


class FinalFieldStatus(StrEnum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    NOT_REPORTED = "not_reported"
    CONFLICTING = "conflicting"
    UNREADABLE = "unreadable"
