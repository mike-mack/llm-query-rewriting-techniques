from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")

decompose_prompt = PromptTemplate.from_template("""
Decompose the following question into a series of smaller, simpler sub-questions 
that should be answered in order from least ot most complex.

Question: "{query}"
""")


query = "How do electric cars affect oil prices and global geopolitics?"

sub_questions = llm.invoke(decompose_prompt.format(query=query)).content
print("Sub-questions:\n", sub_questions)
