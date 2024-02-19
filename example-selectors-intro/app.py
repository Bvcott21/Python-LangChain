from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv()

llm = OpenAI(model_name = 'gpt-3.5-turbo-0125')

# Few shot templates
our_prompt = """
You are a 5 year old girl, who is very funny, mischievous and sweet:

Question: What is a house?
Response:
"""

llm = OpenAI(temperature = .9, model_name = 'gpt-3.5-turbo-0125')
#print(llm(our_prompt))

### Providing examples ###

our_prompt = """
You are a 5 year old girl, who is very funny, mischievous and sweet:

Question: What is a mobile?
Response: A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games and videos.

Question: What are your dreams?
Response: My dreams are like colourful adventures, where I become a superhero and save the day! I dream of giggles and ice cream

Question: What is a house?
Response:
"""

#print(llm(our_prompt))

### FewShotPromptTemplate ###

examples = [
    {
        "query": "What is a mobile",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games and videos."        
    },
    {
        "query": "What are your dreams?",
        "answer": "My dreams are like colourful adventures, where I become a superhero and save the day! I dream of giggles and ice cream."        
    }
]

example_template = """
    Question: {query}
    Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables = ["query", "answer"],
    template = example_template
)

prefix = """You are a 5 year old girl who is very funny, mischievous and sweet:
Here are some examples:
"""

suffix = """
Question: {user_input}
Response: 
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["user_input"],
    example_separator = "\n\n"
)

query = "What is a house?"
#print(few_shot_prompt_template.format(user_input=query))

#print(llm(few_shot_prompt_template.format(user_input=query)))


### Providing even more examples ###

examples = [
    {
        "query": "What is a mobile",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games and videos."        
    },
    {
        "query": "What are your dreams?",
        "answer": "My dreams are like colourful adventures, where I become a superhero and save the day! I dream of giggles and ice cream."        
    },
    {
        "query": "What are your ambitions?",
        "answer": "I wnt to be a super funny comedian, spreading laughter everywhere I go!"
    },
    {
        "query": "What happens when you get sick?",
        "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly and need lots of cuddles."
    },
    {
        "query": "How much do you love your dad?",
        "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top!"
    },
    {
        "query": "Tell me about your friend?",
        "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together."
    },
    {
        "query": "What math means to you?",
        "answer": "Math is like a puzzle game, full of numbers and shapes."
    }, 
    {
        "query": "What is your fear?",
        "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed."
    }
]

example_template = """
    Question: {query}
    Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables = ["query", "answer"],
    template = example_template
)

prefix = """You are a 5 year old girl who is very funny, mischievous and sweet:
Here are some examples:
"""

suffix = """
Question: {user_input}
Response: 
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["user_input"],
    example_separator = "\n\n"
)

query = "What is a house?"
#print(few_shot_prompt_template.format(user_input=query))

#print(llm(few_shot_prompt_template.format(user_input=query)))


### LengthBasedExampleSelector ###

example_selector = LengthBasedExampleSelector(
    examples = examples,
    example_prompt = example_prompt,
    max_length = 1000
)

new_prompt_template = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["user_input"],
    example_separator="\n"
)

query = "What is a house?"
print(new_prompt_template.format(user_input = query))
print(llm(new_prompt_template.format(user_input = query)))

## adding new example to the template ###

new_example = {
    "query": "What's your favourite work?",
    "answer": "Sleep"
}

new_prompt_template.example_selector.add_example(new_example)

example_selector = LengthBasedExampleSelector(
    examples = examples,
    example_prompt = example_prompt,
    max_length = 1000
)

print(new_prompt_template.format(user_input = query))