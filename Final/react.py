import llama_index
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from langchain.chat_models import ChatOpenAI

llm1 = ChatOpenAI(temperature=0.1,model='gpt-3.5-turbo')
llm2 = ChatOpenAI(temperature=0.2,model='gpt-3.5-turbo-0301')
llm3 = ChatOpenAI(temperature=0.1,model='gpt-3.5-turbo')
llm4 = ChatOpenAI(temperature=0.3,model='gpt-3.5-turbo-0301')
llm5 = ChatOpenAI(temperature=0.5,model='gpt-3.5-turbo')
llm6 = ChatOpenAI(temperature=0.4,model='gpt-3.5-turbo-0301')

chp1 = SimpleDirectoryReader(input_dir="data/chp1.txt").load_data()
chp2 = SimpleDirectoryReader(input_dir="data/chp2.txt").load_data()
chp3 = SimpleDirectoryReader(input_dir="data/chp3.txt").load_data()
chp4 = SimpleDirectoryReader(input_dir="data/chp4.txt").load_data()
chp5 = SimpleDirectoryReader(input_dir="data/chp5.txt").load_data()
chp6 = SimpleDirectoryReader(input_dir="data/chp6.txt").load_data()

arr = [chp1,chp2,chp3,chp4,chp5,chp6]
arlm = [llm1,llm2,llm3,llm4,llm5,llm6]

choice = int(input("Select which chapter to choose from ...1 to 6  >>"))-1

service = ServiceContext.from_defaults(llm = arlm[choice],)
index = VectorStoreIndex.from_documents(arr[choice])

chat_eng = index.as_chat_engine(service_context = service, chat_mode='react', verbose=True)

while True:
    question = input("Enter your Question?    >>")
    if question == "Thanks" or question=="End":
        break
    response = chat_eng.chat(question)
    print(response)
    print("\n")