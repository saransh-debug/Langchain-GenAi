from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatGoogleGenerativeAI(model="gemini-2.5-flash" , temperature=1.8 ,
 max_completion_tokens=10)
result = chatmodel.invoke("write a 5 line poem on fifa world cup .")
print(result.content)