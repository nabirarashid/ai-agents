from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from restaurant_agent.vector import retriever

model = OllamaLLM(model="llama3.2")

template= """
    You are an expert in answering questions about a pizza restaurant.
    Here are some relavant reviews: {reviews}
    Here is the question to answer: {question}
    """

prompt = ChatPromptTemplate.from_template(template)

# putting in a chain allows model to automatically use the prompt
chain = prompt | model

while True:
    print("\n\n---------------------------------------------")
    question = input("Ask your question (press 'q' to quit): ")
    print("\n\n---------------------------------------------")
    if question.lower() == "q":
        break

    # retrieve relevant reviews from the vector store
    reviews = retriever.invoke(question)

    # invoke triggers the chain with the provided input
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
