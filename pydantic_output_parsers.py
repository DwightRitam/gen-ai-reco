import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Load environment variables
load_dotenv()

# 2. Define Schema
class FilmRecommendation(BaseModel):
    title: str = Field(description="Name of the movie")
    year: int = Field(description="Release year")
    summary: str = Field(description="A brief one-sentence premise")
    directors: list[str] = Field(description="List of directors")
    actors: list[str] = Field(description="List of top 3 main actors")
    imdb_rating: float = Field(description="IMDb rating out of 10")

# 3. Poster Fetcher
def get_poster_url(movie_title):
    omdb_api_key = os.getenv("OMDB_API_KEY")
    if not omdb_api_key:
        return "https://via.placeholder.com/300x450?text=OMDB+API+Key+Missing"
    
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={omdb_api_key}"
    try:
        response = requests.get(url).json()
        if "Poster" in response and response["Poster"] != "N/A":
            return response["Poster"]
    except Exception:
        pass
    
    return "https://via.placeholder.com/300x450?text=Poster+Not+Found"

# 4. Streamlit UI
st.set_page_config(page_title="Hidden Gem Finder", page_icon="🎬", layout="centered")
st.title("🎬 Hidden Gem Film Finder")
st.markdown("Discover underrated movies with full cast details and posters.")

genre_input = st.text_input("Enter a genre:", placeholder="e.g., Sci-fi, Cyberpunk, Indie Thriller...")

if st.button("Find a Movie", type="primary"):
    if not genre_input:
        st.warning("Please enter a genre.")
    else:
        with st.spinner(f"Scouring the archives for {genre_input} gems..."):
            try:
                # JUST the Pydantic parser
                parser = PydanticOutputParser(pydantic_object=FilmRecommendation)
                
                # Temperature lowered to 0.7 so the JSON formatting doesn't break
                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=1.2
                )

                prompt = ChatPromptTemplate.from_template(
                    """Give me a highly underrated, obscure, or random film suggestion based on this genre: {genre}. Do not suggest popular movies.

                    {format_instructions}

                    CRITICAL: Return ONLY valid JSON. Do not include any conversational text, explanations, or markdown formatting (do NOT use ```json). Your response must start immediately with the opening curly brace {{ and end with }}.
                    """
                )

                # Simple chain
                chain = prompt | llm | parser
                
                result = chain.invoke({
                    "genre": genre_input,
                    "format_instructions": parser.get_format_instructions()
                })

                poster_url = get_poster_url(result.title)

                # 5. Display Columns
                st.success("Match Found!")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(poster_url, use_container_width=True)
                with col2:
                    st.subheader(f"🎥 {result.title} ({result.year})")
                    st.write(f"⭐ **IMDb Rating:** {result.imdb_rating}/10")
                    st.write(f"🎬 **Director(s):** {', '.join(result.directors)}")
                    st.write(f"🎭 **Starring:** {', '.join(result.actors)}")
                    st.write(f"**Premise:** {result.summary}")

            except Exception as e:
                st.error("API Error. Click the button to try again.")
                st.exception(e)