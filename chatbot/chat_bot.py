from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage , HumanMessage , SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []
while True:
    user_input = input("YOU:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break 
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)
print(chat_history)