from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate

load_dotenv()

llm = OpenAI(model_name = 'gpt-3.5-turbo-0125')

our_prompt = """
I love trips, and I have been to 6 countries.
I plan to visit few more soon.

Can you create a post for tweet in 10 words or above?
"""

print(our_prompt)
llm(our_prompt)


# f-string

words_count = 3
our_text = "I love trips, and I have been to 6 countries. I plan to visit few more soon."

our_prompt = f"""
{our_text}

###

Can you create a post for tweet in {words_count} for the above?
"""

print(our_prompt)
llm(our_prompt)



# PromptTemplates

template = """
{our_text}

Can you please craete a post for tweet in {words_count} words for the above?
"""

prompt = PromptTemplate(
    input_variables = ["words_count", "our_text"],
    template = template
)

final_prompt = prompt.format(
    words_count = 3, 
    our_text = "I love trips, and I have been to 6 countries. I plan to visit a few more soon."
)

print(final_prompt)
print(llm(final_prompt))