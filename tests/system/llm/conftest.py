import pytest

from canopy.models.data_models import UserMessage, AssistantMessage


@pytest.fixture
def messages():
    # Create a list of MessageBase objects
    """
    This function returns a list of `MessageBase` objects that represent different
    messages to be used as input for an assistant AI.

    Returns:
        list: The output returned by the `messages()` function is a list of 3
        `UserMessage` objects and 1 `AssistantMessage` object.

    """
    return [
        UserMessage(content="Hello, assistant."),
        AssistantMessage(content="Hello, user. How can I assist you?"),
        UserMessage(content="Just checking in. Be concise."),
    ]
