from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence , RunnablePassthrough , RunnableLambda , RunnableBranch
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace




load_dotenv()

#Runnable lambda is a runnable that takes a function as input and returns a runnable that executes the function when invoked , it is useful when you want to create a runnable that executes a custom function , it can be used in a RunnableSequence or RunnableParallel to execute a custom function as part of the sequence or parallel execution.

temp1 = PromptTemplate(
    template=""" you are a reporter, so based on the topic : {topic} , genrate a report of 5000 words.""" , 
    input_variables=['topic']
)


temp2 = PromptTemplate(
    template=""" summarize this report : {report} , and genrate a response starting with " the summary of the report is : " .""" , 
    input_variables=['report']
)




llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()



runnable = temp1 | model | parser


# def word_count(text):
#     count = 0 
#     for i in text:
#         count+=1
#     return count

parallel_chain = RunnableBranch(
    (lambda x : len(x.split()) < 500 , RunnableSequence(temp2 , model , parser)),
    RunnablePassthrough()
)

final_runnable = RunnableSequence(runnable , parallel_chain)

result = final_runnable.invoke("kids using mobile phones")


print(result)



