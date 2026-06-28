from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

template1 = PromptTemplate(
    template="""give me 5 interesting facts about {country}""",
    input_variables=['country']
)

template2 = PromptTemplate(
    template=""" genrate a summary about the {res} text """ , 
    input_variables=["res"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'country':"India"})

print(result)

chain.get_graph().print_ascii() # to print the flow of the chain 
