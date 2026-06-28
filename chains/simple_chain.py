from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

template = PromptTemplate(
    template="""give me 5 interesting facts about {country}""",
    input_variables=['country']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = template | model | parser 

result = chain.invoke({'country':"India"})

print(result)