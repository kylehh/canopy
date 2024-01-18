import os
from unittest.mock import MagicMock

import jsonschema
import pytest

from canopy.llm import AzureOpenAILLM, AnyscaleLLM
from canopy.models.data_models import Role, MessageBase, Context, StringContextContent  # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk # noqa
from canopy.llm.openai import OpenAILLM  # noqa
from canopy.llm.models import \
    Function, FunctionParameters, FunctionArrayProperty  # noqa
from openai import BadRequestError # noqa

SYSTEM_PROMPT = "You are a helpful assistant."


def assert_chat_completion(response, num_choices=1):
    """
    This function asserts that a chat completion response from a Discord bot
    includes the expected number of choices and each choice has a message with a
    role.

    Args:
        response (): The `response` input parameter is passed as an argument to
            be evaluated and asserted within the function.
        num_choices (int): The `num_choices` parameter specifies the expected
            number of choices present In the response.

    """
    assert len(response.choices) == num_choices
    for choice in response.choices:
        assert isinstance(choice.message, MessageBase)
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert isinstance(choice.message.role, Role)


def assert_function_call_format(result):
    """
    This function checks that the `result` argument is a dictionary with at least
    one query (i.e., a string) inside a list called "queries".

    Args:
        result (dict): The `result` input parameter is not used and can be any
            value because the function only performs assertions on its contents.

    """
    assert isinstance(result, dict)
    assert "queries" in result
    assert isinstance(result["queries"], list)
    assert len(result["queries"]) > 0
    assert isinstance(result["queries"][0], str)
    assert len(result["queries"][0]) > 0


@pytest.fixture
def function_query_knowledgebase():
    """
    This function returns a Function object that represents the "query knowledgebase"
    functionality.

    Returns:
        : The output returned by this function is a `Function` object representing
        the querying of a search engine for relevant information with the provided
        list of queries as input.

    """
    return Function(
        name="query_knowledgebase",
        description="Query search engine for relevant information",
        parameters=FunctionParameters(
            required_properties=[
                FunctionArrayProperty(
                    name="queries",
                    items_type="string",
                    description='List of queries to send to the search engine.',
                ),
            ]
        ),
    )


@pytest.fixture
def model_params_high_temperature():
    """
    The function `model_params_high_temperature()` returns a dictionary with
    parameters for a model assuming high temperature conditions: specifically
    temperatures around 0.9 (i.e., 90% of the maximum possible temperature).

    Returns:
        dict: The output returned by `model_params_high_temperature()` is a
        dictionary with the following keys and values:
        
        { "temperature": 0.9 , "top_p": 0.95 , "n": 3 }

    """
    return {"temperature": 0.9, "top_p": 0.95, "n": 3}


@pytest.fixture
def model_params_low_temperature():
    """
    The given function defines a set of parameter values for a model running at
    low temperatures:
    	- `temperature`: 0.2 (i.e., 200 K)
    	- `top_p`: 0.5 (i.e., the probability of being at the top of the potential
    energy curve is 50%)
    	- `n`: 1 (i.e., the number of samples used to estimate the target distribution)

    Returns:
        dict: The output returned by the function `model_params_low_temperature()`
        is a dictionary with the following parameters:
        
        {"temperature": 0.2,"top_p": 0.5,"n": 1}

    """
    return {"temperature": 0.2, "top_p": 0.5, "n": 1}


@pytest.fixture(params=[OpenAILLM, AzureOpenAILLM, AnyscaleLLM])
def openai_llm(request):
    """
    This function takes a request parameter and returns an instance of a suitable
    LLM (language model) class based on the value of the "llm_class" parameter.
    The possible values for lml_class are AzureOpenAILLM and AnyscaleLLM.

    Args:
        request (): The `request` parameter is not used anywhere within this
            function; it seems to be unused and could potentially be removed without
            any changes to the rest of the code.

    Returns:
        : The output returned by this function is an instance of the LLMCLASS
        (i.e., AzureOpenAILLM or AnyscaleLLM) specified by the request parameter
        with a model name of either "mistralai/Mistral-7B-Instruct-v0.1" for
        AnyscaleLLM or "gpt-3.5-turbo-0613" for all other llm_classes.

    """
    llm_class = request.param
    if llm_class == AzureOpenAILLM:
        model_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        if model_name is None:
            pytest.skip(
                "Couldn't find Azure deployment name. Skipping Azure OpenAI tests."
            )
    elif llm_class == AnyscaleLLM:
        if os.getenv("ANYSCALE_API_KEY") is None:
            pytest.skip("Couldn't find Anyscale API key. Skipping Anyscale tests.")
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    else:
        model_name = "gpt-3.5-turbo-0613"

    return llm_class(model_name=model_name)


def test_init_with_custom_params(openai_llm):
    """
    This function tests the initialisation of an OpenAI LLM instance with custom
    parameters.

    Args:
        openai_llm (): The `openai_llm` input parameter is used to determine if
            the test should skip running or not.

    """
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Tested separately in test_azure_openai.py")

    llm = openai_llm.__class__(
        model_name="test_model_name",
        api_key="test_api_key",
        organization="test_organization",
        temperature=0.9,
        top_p=0.95,
        n=3,
    )

    assert llm.model_name == "test_model_name"
    assert llm.default_model_params["temperature"] == 0.9
    assert llm.default_model_params["top_p"] == 0.95
    assert llm.default_model_params["n"] == 3
    assert llm._client.api_key == "test_api_key"
    assert llm._client.organization == "test_organization"


def test_chat_completion_no_context(openai_llm, messages):
    """
    This function tests the `chat_completion` method of an OpenAI LLM with a random
    set of messages as context.

    Args:
        openai_llm (): The `openai_llm` input parameter is an instance of the
            OpenAI Language Model (LLM) class.
        messages (list): The `messages` input parameter is a list of previous
            messages exchanged during the conversation.

    """
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages)
    assert_chat_completion(response)


def test_chat_completion_with_context(openai_llm, messages):
    """
    This function tests the chat completion functionality of an OpenAI LLM model
    with a given context (content and number of tokens) and a list of previous messages.

    Args:
        openai_llm (): The `openai_llm` parameter is an instance of the OpenAI LLM
            model that is used for chat completion.
        messages (list): The `messages` input parameter provides a list of previous
            messages exchanged during the chat conversation.

    """
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          context=Context(
                                              content=StringContextContent(
                                                  __root__="context from kb"
                                              ),
                                              num_tokens=5
                                          ))
    assert_chat_completion(response)


def test_enforced_function_call(openai_llm,
                                messages,
                                function_query_knowledgebase):
    """
    This function `test_enforced_function_call` tests if the `openai_llm` model
    responds correctly when called with a specific `function_query_knowledgebase`
    function and provides a formatted response as expected.

    Args:
        openai_llm (): The `openai_llm` parameter is an instance of the OpenAI
            LLMCenter class and is used to interact with the language model.
        messages (list): The `messages` input parameter is a list of chat messages
            that the LLaMA model has previously exchanged with the user.
        function_query_knowledgebase (): Based on the name and the context of the
            code snippet you provided:
            
            The `function_query_knowledgebase` parameter is a callable function
            that is used to query the knowledge base of an OpenAI language model
            using the `enforced_function_call()` method.

    """
    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase)
    assert_function_call_format(result)


def test_chat_completion_high_temperature(openai_llm,
                                          messages,
                                          model_params_high_temperature):
    """
    This function tests the chat completion functionality of an OpenAI LLM with a
    given temperature value.

    Args:
        openai_llm (): The `openai_llm` input parameter is an instance of the
            OpenAI Language Model (LLM) class that is being used for chat completion.
        messages (list): The `messages` input parameter is a list of previous
            messages to be used as context for the chat completion.
        model_params_high_temperature (dict): The `model_params_high_temperature`
            input parameter specifies the number of choices for chat completion
            (i.e., the "n" value) when the chat model is running at a high temperature.

    """
    if isinstance(openai_llm, AnyscaleLLM):
        pytest.skip("Anyscale don't support n>1 for the moment.")

    response = openai_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        model_params=model_params_high_temperature
    )
    assert_chat_completion(response,
                           num_choices=model_params_high_temperature["n"])


def test_chat_completion_low_temperature(openai_llm,
                                         messages,
                                         model_params_low_temperature):
    """
    This function tests the chat completion capabilities of an OpenAI LLM model
    at low temperature using the `chat_completion()` method.

    Args:
        openai_llm (): The `openai_llm` input parameter is the OpenAI Language
            Model (LLM) instance to be used for chat completion.
        messages (list): The `messages` input parameter is a list of previous
            messages exchanged between the user and the AI model. It is passed to
            the `openai_llm.chat_completion()` function as part of the chat history.
        model_params_low_temperature (dict): The `model_params_low_temperature`
            input parameter is used to configure the language model's temperature
            (a hyperparameter that controls how speculative the model's predictions
            are) to a low value.

    """
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          model_params=model_params_low_temperature)
    assert_chat_completion(response,
                           num_choices=model_params_low_temperature["n"])


def test_enforced_function_call_high_temperature(openai_llm,
                                                 messages,
                                                 function_query_knowledgebase,
                                                 model_params_high_temperature):
    """
    This function tests the `enforced_function_call` method of an OpenAI LLM (large
    language model) with a system prompt and chat history as input.

    Args:
        openai_llm (): The `openai_llm` parameter is an instance of the OpenAI LLM
            (language learning model) that is being used to perform the enforced
            function call.
        messages (list): In the given function `test_enforced_function_call_high_temperature`,
            the `messages` parameter is a list of chat messages that represent the
            previous interactions between the user and the model.
        function_query_knowledgebase (): In the given function
            `test_enforced_function_call_high_temperature`, the `function_query_knowledgebase`
            input parameter is a callable that represents the function to be called
            with enforced functional constraints.
        model_params_high_temperature (dict): The `model_params_high_temperature`
            input parameter sets the temperature threshold for the LLM's reasoning
            process.

    """
    if isinstance(openai_llm, AnyscaleLLM):
        pytest.skip("Anyscale don't support n>1 for the moment.")

    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase,
        model_params=model_params_high_temperature
    )
    assert isinstance(result, dict)


def test_enforced_function_call_low_temperature(openai_llm,
                                                messages,
                                                function_query_knowledgebase,
                                                model_params_low_temperature):
    """
    This function tests the `enforced_function_call` method of an OpenAI LLM
    (Language Language Model) with a customized set of model parameters at a low
    temperature.

    Args:
        openai_llm (int): Based on the function's implementation and name
            `test_enforced_function_call_low_temperature`, the `openai_llm` parameter
            is an instance of `AnyscaleLLM` and its value will be used to set the
            top probability for the model when calling the `enforced_function_call()`
            method.
        messages (list): The `messages` input parameter is a list of previous chat
            messages sent by the user to the model.
        function_query_knowledgebase (): Based on the function name and parameter
            name "function", it appears that "function_query_knowledgebase" is a
            reference to a Python function that takes no arguments and returns a
            value.
        model_params_low_temperature (dict): The `model_params_low_temperature`
            input parameter sets the model's temperature parameters for a
            low-temperature regime.

    """
    model_params = model_params_low_temperature.copy()
    if isinstance(openai_llm, AnyscaleLLM):
        model_params["top_p"] = 1.0

    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase,
        model_params=model_params
    )
    assert_function_call_format(result)


def test_chat_completion_with_model_name(openai_llm, messages):
    """
    This function tests that the `chat_completion` method of an OpenAI LLM models
    takes a custom model name as a parameter and uses that model for generating
    chat completions.

    Args:
        openai_llm (): The `openai_llm` input parameter is an instance of a OpenAI
            LLM model (either AzureOpenAILLM or AnyscaleLLM), which is used to
            test the chat completion feature with a different model name.
        messages (list): The `messages` input parameter is a list of previous chat
            messages that are used as context for the chat completion.

    """
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("In Azure the model name has to be a valid deployment")
    elif isinstance(openai_llm, AnyscaleLLM):
        new_model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        new_model_name = "gpt-3.5-turbo-1106"

    assert new_model_name != openai_llm.model_name, (
        "The new model name should be different from the default one. Please change it."
    )
    response = openai_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        model_params={"model": new_model_name}
    )

    assert response.model == new_model_name


def test_chat_streaming(openai_llm, messages):
    """
    This function tests the chat streaming functionality of an OpenAI LLM by sending
    a sequence of messages and verifying that the model responds with appropriate
    chat chunks.

    Args:
        openai_llm (): The `openai_llm` input parameter is the instance of the
            OpenAI LLM (language model) that is used to generate responses to the
            user's messages.
        messages (list): The `messages` input parameter is a list of previous chat
            messages that are passed to the `chat_completion` method to simulate
            a conversation history and enable the language model to generate
            coherent responses.

    """
    stream = True
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          stream=stream)
    messages_received = [message for message in response]
    assert len(messages_received) > 0
    for message in messages_received:
        assert isinstance(message, StreamingChatChunk)


def test_max_tokens(openai_llm, messages):
    """
    This function tests that the `chat_completion()` method of an OpenAI LLM model
    returns a response with a maximum number of tokens (2) and that each choice
    returned by the model has a message with a length less than or equal to the
    specified max tokens.

    Args:
        openai_llm (): The `openai_llm` input parameter is an instance of the
            OpenAI language model that is used to generate chat responses.
        messages (list): The `messages` parameter is a list of previous chat
            messages that are passed as input to the `openai_llm.chat_completion()`
            function.

    """
    max_tokens = 2
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          max_tokens=max_tokens)
    assert isinstance(response, ChatResponse)
    assert len(response.choices[0].message.content.split()) <= max_tokens


def test_negative_max_tokens(openai_llm, messages):
    """
    This function tests the `openai_llm` module's `chat_completion` method with a
    negative value for `max_tokens`, which should raise a `RuntimeError`.

    Args:
        openai_llm (): The `openai_llm` parameter is the object of type `OpenAI
            LLM` (Large Language Model) that is being used to generate responses
            to the user's messages.
        messages (list): The `messages` input parameter provides the previous
            messages exchanged between the user and the model for contextual
            understanding and generation of relevant responses.

    """
    with pytest.raises(RuntimeError):
        openai_llm.chat_completion(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            max_tokens=-5)


def test_chat_complete_api_failure_populates(openai_llm,
                                             messages):
    """
    This function tests that an exception is raised when the OpenAI LLM API call
    fails during chat completion.

    Args:
        openai_llm (): The `openai_llm` input parameter is a mock object used to
            replace the actual OpenAI Language Model client object.
        messages (list): The `messages` input parameter is a list of previous chat
            messages that are used to generate a completion suggestion for the
            current user input.

    """
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.side_effect = Exception(
        "API call failed")

    with pytest.raises(Exception, match="API call failed"):
        openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                   chat_history=messages)


def test_enforce_function_api_failure_populates(openai_llm,
                                                messages,
                                                function_query_knowledgebase):
    """
    This function tests that an exception is raised when the API call fails using
    MagicMock and pytest.raises() functions.

    Args:
        openai_llm (): The `openai_llm` input parameter is the object instance of
            the `OpenAI LLMApiClient` class being tested.
        messages (list): The `messages` parameter is used as input for the
            `chat_history` parameter of the `enforced_function_call()` method of
            the `openai_llm` object.
        function_query_knowledgebase (): In this function test_enforce_function_api_failure_populates.
            The `function_query_knowledgebase` parameter is passed as an argument
            to the `openai_llm.enforced_function_call` method which is used to
            simulate a function call.

    """
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.side_effect = Exception(
        "API call failed")

    with pytest.raises(Exception, match="API call failed"):
        openai_llm.enforced_function_call(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          function=function_query_knowledgebase)


def test_enforce_function_wrong_output_schema(openai_llm,
                                              messages,
                                              function_query_knowledgebase):
    """
    This function tests the `enforced_function_call` method of an OpenAI LLM by
    attempting to call the function with an invalid schema.

    Args:
        openai_llm (): The `openai_llm` parameter is a mock object used to test
            the functionality of the `enforced_function_call()` method of the `LLM`
            class.
        messages (list): In the function `test_enforce_function_wrong_output_schema`,
            the `messages` parameter is passed to `openai_llm.enforced_function_call()`
            as a chat history that represents the context of the conversation so
            far.
        function_query_knowledgebase (): In the context of the provided code
            snippet`, the `function_query_knowledgebase` input parameter is passed
            to the `_client.chat.completions.create()` method and is used to define
            the query knowledge base that will be used to retrieve completions for
            the system prompt.

    """
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(
                            arguments="{\"key\": \"value\"}"))]))])

    with pytest.raises(jsonschema.ValidationError,
                       match="'queries' is a required property"):
        openai_llm.enforced_function_call(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          function=function_query_knowledgebase)

    assert openai_llm._client.chat.completions.create.call_count == 3, \
        "retry did not happen as expected"


def test_enforce_function_unsupported_model(openai_llm,
                                            messages,
                                            function_query_knowledgebase):
    """
    This function tests that calling `enforced_function_call()` on an OpenAI LLM
    with an unsupported model (i.e., Azure or Anyscale) raises a `NotImplementedError`.

    Args:
        openai_llm (): The `openai_llm` input parameter is the instance of the
            OpenAI LLM (Language Model) to be used for the call to the
            `enforced_function_call()` method.
        messages (list): The `messages` input parameter is used as the chat history
            for the function call.
        function_query_knowledgebase (): The `function_query_knowledgebase` input
            parameter is a function that retrieves information from the OpenAI
            Knowledge Base using the given model.

    """
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Currently not tested in Azure")
    elif isinstance(openai_llm, AnyscaleLLM):
        new_model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        new_model_name = "gpt-3.5-turbo-0301"

    with pytest.raises(NotImplementedError):
        openai_llm.enforced_function_call(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            function=function_query_knowledgebase,
            model_params={"model": new_model_name}
        )


def test_available_models(openai_llm):
    """
    This function checks that the `available_models` method of an OpenAI LLM
    instance (either locally installed or Azure-based) returns a list of at least
    one model name (as strings), and that the current LLM instance's name is
    included within that list.

    Args:
        openai_llm (): The `openai_llm` input parameter is an instance of the
            AzureOpenAILLM class.

    """
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Azure does not support listing models")
    models = openai_llm.available_models
    assert isinstance(models, list)
    assert len(models) > 0
    assert isinstance(models[0], str)
    assert openai_llm.model_name in models


@pytest.fixture()
def no_api_key():
    """
    This function `no_api_key` undoes the effects of a previous call to
    `os.environ["OPENAI_API_KEY"]`, by temporarily storing and then restoring the
    previous value of the environment variable "OPENAI_API_KEY".

    """
    before = os.environ.pop("OPENAI_API_KEY", None)
    yield
    if before is not None:
        os.environ["OPENAI_API_KEY"] = before


def test_missing_api_key(no_api_key):
    """
    This function tests that the `OpenAILLM` class raises a `RuntimeError` when
    initialized without an `OPENAI_API_KEY` environment variable set.

    Args:
        no_api_key (str): The `no_api_key` input parameter is not passed to the
            `OpenAILLM()` constructor.

    """
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLM()


@pytest.fixture()
def bad_api_key():
    """
    This function temporarily replaces the `OPENAI_API_KEY` environment variable
    with a bad value and then yields control to the next expression.

    """
    before = os.environ.pop("OPENAI_API_KEY", None)
    os.environ["OPENAI_API_KEY"] = "bad key"
    yield
    if before is not None:
        os.environ["OPENAI_API_KEY"] = before


def test_bad_api_key(bad_api_key, messages):
    """
    This function tests the `OpenAILLM` class's `chat_completion` method with a
    bad API key and raises a `RuntimeError` with a message that includes "API key".

    Args:
        bad_api_key (str): The `bad_api_key` input parameter passes a wrong or
            invalid API key to the function test_bad_api_key(), which triggers an
            exception with the message "API key".
        messages (list): In the given function `test_bad_api_key`, the `messages`
            input parameter is used to provide a list of previous messages exchanged
            between the user and the chatbot.

    """
    with pytest.raises(RuntimeError, match="API key"):
        llm = OpenAILLM()
        llm.chat_completion(system_prompt=SYSTEM_PROMPT, chat_history=messages)
