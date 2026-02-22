from dataclasses import dataclass

@dataclass
class Email:
    """Dataclass representing an email with subject and body"""
    subject: str
    body: str

@dataclass
class TopicDiscription:
    """Dataclass representing an email with topic and description"""
    description: str