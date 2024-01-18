from typing import Optional, Any
import os
from canopy.llm import OpenAILLM
from canopy.llm.models import Function
from canopy.models.data_models import Messages

ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"
FUNCTION_MODEL_LIST = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


class AnyscaleLLM(OpenAILLM):
    """
    Anyscale LLM wrapper built on top of the OpenAI Python client.

    Note: Anyscale requires a valid API key to use this class.
          You can set the "ANYSCALE_API_KEY" environment variable.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        *,
        base_url: Optional[str] = ANYSCALE_BASE_URL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        This function initiates an object of a class with a specific model name
        and sets the Anyscale API key and base URL using environment variables or
        provided arguments.

        Args:
            model_name ("meta-llama/Llama-2-7b-chat-hf"): The `model_name` parameter
                is the name of the Anyscale model to be used.
            base_url (ANYSCALE_BASE_URL): The `base_url` input parameter specifies
                the base URL for Anyscale's API endpoints. It defaults to
                `ANYSCALE_BASE_URL`, an environment variable or a configuration
                value that is not set here. If neither is provided and the `api_key`
                parameter is absent or invalid as well., then a ValueError will
                be raised .
            api_key (None): The `api_key` input parameter is used to specify the
                Anyscale API key required for communication with the Anyscale platform.
            	-*kwargs (Any): The **kwargs input parameter is a keyword-only parameter
                that allows passing additional parameters to the super().__init()
                method.

        """
        ae_api_key = api_key or os.environ.get("ANYSCALE_API_KEY")
        if not ae_api_key:
            raise ValueError(
                "Anyscale API key is required to use Anyscale. "
                "Please provide it as an argument "
                "or set the ANYSCALE_API_KEY environment variable."
            )
        ae_base_url = base_url
        super().__init__(model_name, api_key=ae_api_key, base_url=ae_base_url, **kwargs)

    def enforced_function_call(
        self,
        system_prompt: str,
        chat_history: Messages,
        function: Function,
        *,
        max_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> dict:
        """
        This function checks if the specified model supports function calling and
        if it is enabled for the current system prompt before allowing the function
        call to proceed.

        Args:
            system_prompt (str): The `system_prompt` parameter is the text that
                prompts the user to enter a response.
            chat_history (Messages): The `chat_history` parameter is used to pass
                the previous messages exchanged between the user and the assistant
                to the function being called.
            function (Function): The `function` parameter is an optional function
                object that will be called with no arguments and its return value
                passed as the actual function call result.
            max_tokens (None): The `max_tokens` parameter limits the number of
                tokens that can be returned by the function call.
            model_params (None): The `model_params` input parameter is an optional
                dictionary that can contain information about the specific instance
                of the model being used.

        Returns:
            dict: The output returned by this function is `dict`.

        """
        model = self.model_name
        if model_params and "model" in model_params:
            model = model_params["model"]
        if model not in FUNCTION_MODEL_LIST:
            raise NotImplementedError(
                f"Model {model} doesn't support function calling. "
                "To use function calling capability, please select a different model.\n"
                "Pleaes check following link for details: "
                "https://docs.endpoints.anyscale.com/guides/function-calling"
            )
        else:
            return super().enforced_function_call(
                system_prompt, chat_history, function,
                max_tokens=max_tokens, model_params=model_params
            )

    def aenforced_function_call(self,
                                system_prompt: str,
                                chat_history: Messages,
                                function: Function,
                                *,
                                max_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None
                                ):
        """
        This function `aenforced_function_call` raises a `NotImplementedError`.

        Args:
            system_prompt (str): The `system_prompt` input parameter is not
                used/consumed inside the `aenforced_function_call()` function as
                per its implementation.
            chat_history (Messages): The `chat_history` parameter is an instance
                of the `Messages` class containing a sequence of previous messages
                sent and received during the conversation with the user.
            function (Function): The `function` parameter is an optional input
                that allows the `aenforced_function_call` function to call any
                Python function of the user's choice.
            max_tokens (None): The `max_tokens` input parameter specifies the
                maximum number of tokens (i.e., input items) to use when invoking
                the function with the chat history and system prompt.
            model_params (None): The `model_params` input parameter is an optional
                dictionary of model parameters that can be passed to the `function`
                being called.

        """
        raise NotImplementedError()
