# All your imports here...

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_response(retriever, question):
    # Define the LLM
    model_id = "google/flan-t5-base"
    hf_pipeline = pipeline("text2text-generation", model=model_id, tokenizer=model_id)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Define the prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    # Define the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    # Invoke the chain with the user's question and return the result
    response = rag_chain.invoke(question)
    return response