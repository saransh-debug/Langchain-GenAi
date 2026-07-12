from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence , RunnablePassthrough , RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace




load_dotenv()

#Runnable lambda is a runnable that takes a function as input and returns a runnable that executes the function when invoked , it is useful when you want to create a runnable that executes a custom function , it can be used in a RunnableSequence or RunnableParallel to execute a custom function as part of the sequence or parallel execution.

temp1 = PromptTemplate(
    template=""" you are a comdeian , so based on the topic : {topic} , create a joke  .""" , 
    input_variables=['topic']
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


# def word_count(text):
#     count = 0 
#     for i in text:
#         count+=1
#     return count

parallel_chain = RunnableParallel(
    {
        "joke":RunnablePassthrough(),
        "total_words":RunnableLambda(lambda x : len(x.split()))
    }
)

main_runnable = RunnableSequence(runnable , parallel_chain)

result = main_runnable.invoke("books")

print("joke :",result['joke'])
print("total words in the joke :",result['total_words'])



