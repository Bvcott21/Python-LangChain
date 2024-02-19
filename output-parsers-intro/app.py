from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema

load_dotenv()

llm = OpenAI(temperature = .9, model_name = 'gpt-3.5-turbo-0125')

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template = "Provide 5 examples of {query}.\n{format_instructions}",
    input_variables = ["query"],
    partial_variables = {"format_instructions": format_instructions}
)

prompt = prompt.format(query = "Currencies")

print(prompt)

output = llm(prompt)
print(output)


### JSON Format ###


response_schemas = [
    ResponseSchema(name = "currency", description="answer to the user's question"),
    ResponseSchema(name = "abbreviation", description = "What's the abbreviation of that currency?")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

print(output_parser)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = PromptTemplate(
    template = "Answer the users questions as best as possible.\n{format_instructions}\n{query}",
    input_variables = ["query"],
    partial_variables = {"format_instructions": format_instructions}
)

print(prompt)
prompt = prompt.format(query = "What's the currency of India?")

output = llm(prompt)
print(output)