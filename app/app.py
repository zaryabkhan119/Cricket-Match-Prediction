import os
import streamlit as st
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.getcwd(), ".env")  # Local path
load_dotenv(env_path)

# ✅ Check if API Key is Loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: OpenAI API Key is missing. Please check your .env file.")
    raise ValueError("OpenAI API Key is missing!")

# ✅ Load OpenAI API Key
client = OpenAI(api_key=api_key)

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="chroma_db")  # Ensure correct local path
collection = chroma_client.get_collection(name="cricket_insights")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)

# Query Function
def query_llm(user_query, top_k=3):
    try:
        # ✅ Encode user query
        query_embedding = embedding_model.encode(user_query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        if not results["metadatas"] or not results["metadatas"][0]:
            return "❌ No relevant insights found. Try rephrasing your query."

        # ✅ Extract insights
        retrieved_insights = [result["text"] for result in results["metadatas"][0]]

        # ✅ Soft Filtering: Prioritize Relevant Insights but Keep Some Context
        primary_insights = [insight for insight in retrieved_insights if any(word in insight.lower() for word in user_query.lower().split())]
        if not primary_insights:
            primary_insights = retrieved_insights  # Fall back on general insights

        combined_insights = "\n".join(primary_insights)

        # ✅ Improved Prompt (Less Restrictive)
        prompt = f"""
        You are a cricket expert AI. **Use the facts below to answer the user's question as accurately as possible.**
        If the answer is unclear, provide the best possible explanation.

        **Retrieved Facts:**
        {combined_insights}

        **User Question:** {user_query}

        **Fact-based answer with some reasoning. Don't ignore useful context.**
        """

        # ✅ Generate Response using GPT-4 (Balanced Temperature)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a cricket expert providing highly accurate answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5  # ✅ Allow a bit more flexibility
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Streamlit Chatbot UI
st.set_page_config(page_title="Cricket AI Chatbot", page_icon="🏏")

st.title("🏏 Cricket Insights Chatbot")
st.markdown("Ask me anything about cricket stats, team performance, venues, and player records!")

# User Query Input
user_query = st.text_input("🔎 Enter your cricket-related question:")

if st.button("Ask AI"):
    if user_query:
        with st.spinner("Fetching Insights..."):
            gpt_response = query_llm(user_query)
        st.markdown(f"### 🤖 AI's Response:")
        st.write(gpt_response)
    else:
        st.warning("⚠️ Please enter a question.")

st.markdown("---")
st.markdown("🔹 **Powered by GPT-4 and ChromaDB**")
