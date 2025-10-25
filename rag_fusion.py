"""
RAG Fusion demo using LangChain + OpenAI.

What this does:
- Generates multiple alternative phrasings of a user question.
- Retrieves results for each query independently.
- Combines the ranked lists using Reciprocal Rank Fusion (RRF).
- Uses the fused top results as context to answer the question.

Notes:
- Uses Chroma vector store (no faiss dependency).
- Uses OpenAIEmbeddings from langchain_openai and a modern embedding model.
- Validates OPENAI_API_KEY via .env or environment variable.
"""

import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate


# Setup and safety 
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
	raise RuntimeError(
		"OPENAI_API_KEY not found. Set it in your environment or a .env file."
	)


# Sample corpus 
texts = [
	"Paris is the capital of France and known for its art and fashion.",
	"Berlin is the capital of Germany and a major tech hub in Europe.",
	"Tokyo, Japan's capital, is famous for its technology and cuisine.",
	"Seoul, the capital of South Korea, is known for K-pop, cutting-edge electronics, and vibrant street food culture.",
	"Paris hosts the Eiffel Tower, one of the most visited monuments in the world.",
]


# Vector store & LLM 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embedding=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini")


# Utility: normalize LLM content to string 
def _content_to_text(content) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts = []
		for p in content:
			if isinstance(p, dict) and "text" in p:
				parts.append(p.get("text", ""))
			else:
				parts.append(str(p))
		return "\n".join([s for s in parts if s])
	return str(content)


# Step 1: Generate alternative queries 
multiquery_prompt = PromptTemplate.from_template(
	"""
You are helping improve information retrieval.
Given the question: "{query}"
Generate 5 alternative phrasings that might retrieve different relevant information.
Output each on a new line.
"""
)

user_query = "What city has the Eiffel Tower?"
mq_response = llm.invoke(multiquery_prompt.format(query=user_query))
mq_text = _content_to_text(mq_response.content)
alt_queries = [user_query] + [q.strip() for q in mq_text.splitlines() if q.strip()]

print("Generated Queries:")
for q in alt_queries:
	print(" -", q)


# Step 2: Retrieve for each query 
ranked_lists = []  # list of lists; inner list is ranked docs for that query
top_k_per_query = 3
for q in alt_queries:
	docs = vectorstore.similarity_search(q, k=top_k_per_query)
	ranked_lists.append(docs)


# Step 3: Reciprocal Rank Fusion (RRF) 
def rrf_fuse(ranked_lists, k: int = 60, top_n: int = 3):
	scores = defaultdict(float)  # key: page_content, value: fused score
	doc_by_key = {}
	for ranked in ranked_lists:
		for rank, d in enumerate(ranked, start=1):
			key = d.page_content
			doc_by_key[key] = d
			scores[key] += 1.0 / (k + rank)
	# sort by score desc
	fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	# return top docs
	return [doc_by_key[key] for key, _ in fused[:top_n]]


fused_docs = rrf_fuse(ranked_lists, k=60, top_n=3)

print("\nFused Documents (RRF):")
for d in fused_docs:
	print("-", d.page_content)


# Step 4: Read/Answer using fused context 
context = "\n".join([d.page_content for d in fused_docs])
answer_prompt = PromptTemplate.from_template(
	"""
Use the fused context below to answer the question.

Context:
{context}

Question:
{query}
"""
)

final_answer = llm.invoke(answer_prompt.format(context=context, query=user_query))
print("\nFinal Answer:\n", final_answer.content)
