from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence , RunnablePassthrough
from pydantic import BaseModel , Field 
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal


load_dotenv()

# basically the runnable sequence can be written as | (pipe operator) between two or more then two runnables , or it can also be used with the Runnable Sequence class which takes a list of runnables as input and executes them in sequence , the output of the first runnable is passed as input to the second runnable and so on , the final output is returned as the output of the RunnableSequence class. its syntax is RunnableSequence([runnable1 , runnable2 , runnable3 , ...]) or runnable1 | runnable2 | runnable3 | ... , both are same and can be used interchangeably.

temp1 = PromptTemplate(
    template=""" you are a rapper , so based on the topic : {topic} , write an rap of 100 words or approx and the rap verse should contain some hidden and double meaning references in the verse that are not that much easier to understand in the first look .""" , 
    input_variables=['topic']
)


temp2 = PromptTemplate(
    template=""" you are an bar-decoder , who decode the bars and the hidden references in a rap , so here is a rap verse {verse} , give me 5 most effiective hidden bars , and with their meaning""" , 
    input_variables=['verse']

)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# chain = temp1 | model | parser 

# result1  = chain.invoke("nature")
# chain2 =  temp2 | model | parser

# result2 = chain2.invoke(result1)


runnable = RunnableSequence(
    first= temp1 , 
    middle=[
        model , 
        parser , 
        temp2 , 
        model 
    ],
    last=parser,
)

result1 = runnable.invoke("gangsters")
print(result1)

