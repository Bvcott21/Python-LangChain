from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from dotenv import load_dotenv
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
import tiktoken
from langchain.memory import ConversationTokenBufferMemory

load_dotenv()

llm = OpenAI(temperature = .9, model_name = 'gpt-3.5-turbo-0125')

### ConversationBufferMemory ###

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    memory = ConversationBufferMemory()
)

# print(conversation.prompt.template)

# conversation("Good morning AI!")
# conversation("My name is Sharath!")
# conversation.predict(input = "I stay in Caracas, Venezuela")

# print(conversation.memory.buffer)

### ConversationBufferWindowMemory
print("\n\n\n --- ConversationBufferWindowMemory --- \n\n\n")

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    # ConversationBufferWindowMemory can take a parameter of 'k' 
    # which will dictate how many responses are stored in memory 
    # I. E. ConversationBufferWindowMemory(k=3) will store the last three responses
    memory = ConversationBufferWindowMemory() 
)

print(conversation.prompt.template)
# conversation.predict(input="Good morning AI!!")
# conversation.predict(input="My name is Edgar")
# conversation.predict(input="I stay in Caracas, Venezuela")
# conversation.predict(input="What is my name?")

# print(conversation.memory.buffer)

### ConversationSummaryMemory ###

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    memory = ConversationSummaryMemory(llm = llm)
)

print(conversation.memory.prompt.template)

conversation.predict(input="good morning AI!")
conversation.predict(input="My name is Edgar")
conversation.predict(input="I stay in Caracas, Venezuela")
print(conversation.memory.buffer)