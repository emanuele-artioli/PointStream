from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.shared.schemas import EncodedChunkPayload

T = TypeVar("T")


class BaseTransport(ABC):
    @abstractmethod
    def send(self, payload: EncodedChunkPayload) -> None:
        """Sends encoded chunk payload to the target medium."""

    @abstractmethod
    def receive(self, chunk_id: str) -> EncodedChunkPayload:
        """Receives encoded chunk payload from the target medium."""


class BaseProcessor(ABC, Generic[T]):
    @abstractmethod
    def process(self, *args, **kwargs) -> T:
        """Runs one processing step and returns structured output."""
