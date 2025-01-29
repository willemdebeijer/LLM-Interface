from abc import ABC, abstractmethod


# Use ABC instead of Protocol since Protocol is not compatible with pydantic BaseModel
class LlmRepresentable(ABC):
    @property
    @abstractmethod
    def llm_repr(
        self,
    ) -> str:
        """Return a string representation suitable for LLM consumption"""
        raise NotImplementedError(
            f"Class {self.__class__.__name__} must implement llm_repr property"
        )
