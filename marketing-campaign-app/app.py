import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv()

def get_llm_response(query, age_option, tasktype_option):
    llm = OpenAI(temperature = .9, model_name = 'gpt-3.5-turbo-0125')

    if age_option == 'Kid':
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
    
    elif age_option == 'Adult':
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
    
    elif age_option == 'Senior Citizen':
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

    prefix = """You are a {template_age_option}, and {template_tasktype_option}:
    Here are some examples:
    """

    suffix = """
    Question: {template_user_input}
    Response: 
    """
    example_selector = LengthBasedExampleSelector(
        examples = examples,
        example_prompt = example_prompt,
        max_length = 200
    )

    new_prompt_template = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = prefix,
        suffix = suffix,
        input_variables = ["template_user_input", "template_age_option", "template_tasktype_option"],
        example_separator = "\n"
    )
    
    print(new_prompt_template.format(
        template_user_input = query, 
        template_age_option = age_option, 
        template_tasktype_option=tasktype_option)
    )

    response = llm(new_prompt_template.format( template_user_input = query, 
        template_age_option = age_option, 
        template_tasktype_option=tasktype_option
        ))
    
    return response

### UI Starts Here ###

st.set_page_config(
    page_title = "Marketing Tool",
    page_icon = ':robot:',
    layout = 'centered',
    initial_sidebar_state = 'collapsed'
)

st.header("Hey, How can I help you?")

form_input = st.text_area('Enter text', height = 275)

tasktype_option = st.selectbox(
    'Please select the action to be performed',
    ('Write a sales copy', 'Create a tweet', 'Write a product description'),
    key = 1
)

age_option = st.selectbox(
    'For which age group?',
    ('Kid', 'Adult', 'Senior Citizen'),
    key=2
)

number_of_words = st.slider('Words limit', 1, 200, 25)

submit = st.button('Generate')

### UI Ends Here ###

if submit:
    st.write(get_llm_response(form_input, age_option, tasktype_option))