import pytest

from llm_interface.helpers import LlmConversionHelpers


@pytest.mark.asyncio
async def test_function_to_tool_multiline_docstring_and_parameters():
    """Test that we can convert a function to a tool with a multiline docstring and parameters"""

    docstring_short = "Get the current temperature for a given location."
    docstring_additional = "This is a multiline docstring."
    param_1_first_line = "City e.g. Bogotá, Colombia"
    param_1_additional = "This should be included"
    param_2_first_line = "Country e.g. Colombia"
    param_2_additional = "This should also be included"

    def get_weather(city: str, country: str) -> str:
        """Get the current temperature for a given location.

        This is a multiline docstring.

        :param city: City e.g. Bogotá, Colombia
            This should be included
        :param country: Country e.g. Colombia
            This should also be included
        """
        return "25 degrees Celsius"

    tool = LlmConversionHelpers.function_to_tool(get_weather)
    assert (
        tool["function"]["description"]
        == f"{docstring_short}\n\n{docstring_additional}"
    )
    assert (
        tool["function"]["parameters"]["properties"]["city"]["description"]
        == f"{param_1_first_line}\n{param_1_additional}"
    )
    assert (
        tool["function"]["parameters"]["properties"]["country"]["description"]
        == f"{param_2_first_line}\n{param_2_additional}"
    )
