import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Configure the Streamlit page UI
st.set_page_config(
    page_title="Filmotrend Recommender", 
    page_icon="🍿", 
    layout="centered"
)

st.title("🍿 Filmotrend Recommender")
st.markdown("Get high-speed movie suggestions powered by Groq and LangChain.")

# 3. Create the user input field
genre_input = st.text_input(
    "What genre are you in the mood for?", 
    placeholder="e.g., Cyberpunk, Psychological Horror, Indie Romance..."
)

# 4. Trigger the LangChain pipeline when the button is clicked
if st.button("Generate Suggestion", type="primary"):
    if not genre_input:
        st.warning("Please enter a genre first.")
    else:
        # Show a loading spinner while the API fetches the output
        with st.spinner("Generating recommendation..."):
            try:
                # Initialize the model
                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.7
                )
                
                # Build the chain
                prompt = ChatPromptTemplate.from_template(
                    "Give me a film suggestion based on this genre: {genre}. Include a brief premise."
                )
                parser = StrOutputParser()
                chain = prompt | llm | parser
                
                # Execute the chain
                result = chain.invoke({"genre": genre_input})
                
                # Display the clean string result
                st.success("Recommendation Found!")
                st.write(result)
                
            except Exception as e:
                st.error(f"API Error: {e}")