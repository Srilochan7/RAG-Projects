import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
from utils.search import duckDuckSearch
from utils.scrapper import scrape_page
from langchain.prompts import PromptTemplate
p = StrOutputParser()

st.set_page_config(page_title="AI Research Agent for Podcasts")
st.title("Recast-AI")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter GROQ API Key")
    st.stop()

name = st.text_input(label="Enter your Guest ?")

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

button = st.button(label="Research")

if button and name:
    with st.spinner("Analyzing about your guest..."):
        links = duckDuckSearch(name=name)

        # st.write("🔗 Links Fetched:")
        # st.write(links)

        data = ""
        for link in links:
            scraped = scrape_page(link)
            data += scraped + "\n\n"

        # st.write("📄 Data Scraped:")
        # st.write(data[:1500]) 
        
        


        prompt = PromptTemplate(
            template="""
        You are an AI-powered podcast research analyst. The user is about to interview or talk about a person named "{name}".
        Use the following scraped content and online data to provide deep, current, and engaging insights. Your output should sound like a research analyst prepping a podcast host — well-structured, smart, and full of value beyond surface-level info.

        Scraped Content:
        {context}

        Structure the output with the following sections:

        🔹 **Quick Bio**  
        Summarize who this person is in 2–3 impactful sentences.

        🔹 **Notable Achievements**  
        Mention major milestones, awards, or known contributions — highlight anything that would make them stand out.

        🔹 **Recent Developments (Last 1-2 Years)**  
        Fetch and summarize their latest work, news, launches, or controversies. This helps the podcast feel current.

        🔹 **Hidden or Lesser-Known Facts**  
        Include 2–3 facts that aren’t commonly known but are interesting for storytelling.

        🔹 **Smart Podcast Intro Line**  
        Write a one-liner the host can use to introduce them with flair — should sound informed and punchy.

        🔹 **Potential Topics to Ask About**  
        List 3–4 conversation angles the host could take based on the data — must sound intelligent and fresh.

        Make the response sound like a podcast producer’s briefing document. Make it smart, concise, and engaging.
        """,
            input_variables=["name", "context"]
        )


        
        chain = prompt | model | p
        
        result = chain.invoke({'name': name, 'context':data})
        
        
        st.write(result)