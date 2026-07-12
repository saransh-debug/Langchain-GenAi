from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence

load_dotenv()

temp1 = PromptTemplate(
    template=""" give me the best pc games of all times , and rank them from 1 to 10 , based on their average rating , on the genre provided {genre} , the output format should be like ( rank 1 : game name  , rating : game rating ) in JSON format """ , 
    input_variables=['genre']
)

temp2 = PromptTemplate(
    template=""" give me the worst pc games of all times , based on the average rating creteria , on this genre:{genre}
     this is the output format ( rank 1 : game name  , rating : game rating ) in JSON format"""
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = JsonOutputParser()

parallel_chain = RunnableParallel(
    {
        "best":RunnableSequence(first=temp1 , 
                                middle=[model] , 
                                last=parser) , 
        "worst":RunnableSequence(
            first=temp2 , 
            middle=[model] , 
            last = parser
        )
    }

)

result = parallel_chain.invoke({"open world"})

print(result)

