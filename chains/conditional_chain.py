from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field 
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal


load_dotenv()




class sentiment_analysis(BaseModel):
    sentiment : Literal['positive' , 'negetive'] =Field(description="give the sentiment analysis of the input text ")

py_parser = PydanticOutputParser(pydantic_object=sentiment_analysis)

template1 = PromptTemplate(
    template="""do the sentiment analysis of the following feedback  {feedback} \n {format_instruction} """ , 
    input_variables=['feedback'] , 
    partial_variables={'format_instruction':py_parser.get_format_instructions()} # adds the instruction about the structure of the inst required into the prompt .
)


parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

classifier_chain = template1 | model | py_parser


template2 = PromptTemplate(
    template=""" you are an feedback agent , so based on the sentiment {sentiment} , give an single appropriate feedback to the customer.""" , 
    input_variables=['sentiment']
)



Runnable_branch = RunnableBranch(
    (lambda x : x.sentiment=='positive' , template2 | model | parser) , 
    (lambda x : x.sentiment=='negetive' , template2 | model | parser) , 
    RunnableLambda(lambda x : "count not find any kind of sentiment ")
)

chain = classifier_chain | Runnable_branch 

print(chain.invoke({"feedback":"this is a bad smartphone."}))

chain.get_graph().print_ascii()