from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence , RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser



load_dotenv()

#Runnable passthrough is a runnable that takes an input and returns the same input as output without any modification , it is useful when you want to pass the output of one runnable to another runnable without any modification , it can be used in a RunnableSequence or RunnableParallel to pass the output of one runnable to another runnable without any modification .

temp1 = PromptTemplate(
    template=""" you are a comdeian , so based on the topic : {topic} , create a joke  .""" , 
    input_variables=['topic']
)


temp2 = PromptTemplate(
    template=""" you are an joke-decoder , who decode the jokes and the hidden references in a joke , so here is a joke  {joke} , explain this joke to others.""" , 
    input_variables=['joke']

)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()



runnable = RunnableSequence( # we can also write the pipe version of it also 
    first= temp1 , 
    middle=[
        model 
    ],
    last=parser,
)

parallel_chain = RunnableParallel(
    {
        "joke":RunnablePassthrough(),
        "explaination":RunnableSequence(
            first=temp2 , 
            middle=[model] , 
            last=parser
        )
    }
)

main_runnable = RunnableSequence(runnable , parallel_chain)

result = main_runnable.invoke("books")

print(result['joke'])
print(result['explaination'])



