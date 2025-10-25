from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


step_back_prompt = PromptTemplate.from_template("""
You are helping with information retrieval. The user asks: "{query}"
First, step back and describe the broader topic or question this relates to. 
Then output ONLY the general version of the query.
""")


query = "What are the health benefits of green tea?"
general_query = llm.invoke(step_back_prompt.format(query=query)).content

print("General Query:", general_query)