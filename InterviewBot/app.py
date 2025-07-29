import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from utils import process_file
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI-ML Interview Bot")

s = StrOutputParser()

st.title("🤖 AI-ML Interview Bot")

groq_api_key = st.text_input(label="Enter your Groq API key:", type="password")

n = st.slider(label="Choose the number of questions:", min_value=1, max_value=10)

if st.button("Start Test"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key to start your test.")
        st.stop()

    st.session_state["test_started"] = True
    st.session_state["current_q"] = 0
    st.session_state["score_texts"] = []
    st.session_state["questions"] = []

    vector_store = process_file('questions.txt')
    retriver = vector_store.as_retriever()
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriver,
    )

    for _ in range(n):
        response = chain.invoke({"query": "Give me a relevant question from the document randomly."})
        st.session_state["questions"].append(response['result'])

# Run test if started
if st.session_state.get("test_started", False):
    q_index = st.session_state["current_q"]
    if q_index < n:
        question = st.session_state["questions"][q_index]
        st.markdown(f"**Question {q_index + 1}:** {question}")
        user_answer = st.text_input("Your answer:", key=f"user_answer_{q_index}")

        if st.button("Submit Answer", key=f"submit_{q_index}"):
            mt = PromptTemplate(
                template="""
                Here is a question: {question}
                Here is the user's answer: {user_answer}
                Evaluate the correctness of the answer on a scale of 0 to 1, and explain briefly
                """,
                input_variables=['question', 'user_answer'],
            )

            marks_chain = mt | ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key) | s
            res = marks_chain.invoke({'question': question, 'user_answer': user_answer})
            st.session_state["score_texts"].append(res)
            st.session_state["current_q"] += 1
            st.rerun()
    else:
        st.subheader("✅ Test Completed")
        for i, score in enumerate(st.session_state["score_texts"], 1):
            st.markdown(f"**Q{i} Feedback:** {score}")
        st.success("🎯 Your Total Feedback is shown above.")
