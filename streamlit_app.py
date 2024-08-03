import streamlit as st
from docx import Document
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
import sqlite3
import pandas as pd
from difflib import SequenceMatcher

# Access your API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "college-buddy"

# Initialize OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# List of example questions
EXAMPLE_QUESTIONS = [
    "What are the steps to declare a major at Texas Tech University",
    "What are the GPA and course requirements for declaring a major in the Rawls College of Business?",
    "How can new students register for the Red Raider Orientation (RRO)",
    "What are the key components of the Texas Tech University Code of Student Conduct",
    "What resources are available for students reporting incidents of misconduct at Texas Tech University",
    "What are the guidelines for amnesty provisions under the Texas Tech University Code of Student Conduct",
    "How does Texas Tech University handle academic misconduct, including plagiarism and cheating",
    "What are the procedures for resolving student misconduct through voluntary resolution or formal hearings",
    "What are the rights and responsibilities of students during the investigative process for misconduct at Texas Tech University",
    "How can students maintain a healthy lifestyle, including nutrition and fitness, while attending Texas Tech University"
]

# Initialize SQLite database
@st.cache_resource
def get_database_connection():
    conn = sqlite3.connect('college_buddy.db', check_same_thread=False)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, links TEXT)''')
    conn.commit()

def insert_document(id, title, tags, links):
    if tags.strip() and links.strip():
        conn = get_database_connection()
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO documents (id, title, tags, links) VALUES (?, ?, ?, ?)",
                  (id, title, tags, links))
        conn.commit()
    else:
        st.warning(f"Document '{title}' not inserted due to empty tags or links.")

def get_all_documents():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM documents WHERE tags != '' AND links != ''")
    return c.fetchall()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, file_name, file_id):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "file_name": file_name,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_text": chunk
        }
        index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
        time.sleep(1)  # To avoid rate limiting

# Function to query Pinecone
def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

def generate_related_keywords(text):
    keyword_prompt = f"Generate 5-10 relevant keywords or phrases from this text, separated by commas: {text}"
    keyword_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a keyword extraction assistant. Generate relevant keywords or phrases from the given text."},
            {"role": "user", "content": keyword_prompt}
        ]
    )
    keywords = keyword_response.choices[0].message.content.strip().split(',')
    return [keyword.strip() for keyword in keywords]

def query_db_for_keywords(keywords):
    conn = get_database_connection()
    c = conn.cursor()
    query = """
    SELECT DISTINCT id, title, tags, links 
    FROM documents 
    WHERE tags LIKE ?
    """
    results = []
    for keyword in keywords:
        c.execute(query, (f'%{keyword}%',))
        for row in c.fetchall():
            score = sum(SequenceMatcher(None, keyword.lower(), tag.lower()).ratio() for tag in row[2].split(','))
            results.append((score, row))
    
    # Sort by score in descending order and return the top result
    results.sort(reverse=True, key=lambda x: x[0])
    return results[0][1] if results else None

# Function to get answer from GPT-3.5-turbo
def get_answer(query):
    context = query_pinecone(query)
    max_context_tokens = 3000
    truncated_context = truncate_text(context, max_context_tokens)
    
    # Generate keywords from the query
    query_keywords = generate_related_keywords(query)
    
    # Generate an initial response based on the query and context
    initial_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are College Buddy, an advanced AI assistant designed to help students with their academic queries. Your primary function is to analyze and provide insights based on the context of uploaded documents."""},
            {"role": "user", "content": f"Context: {truncated_context}\n\nQuestion: {query}"}
        ]
    )
   
    initial_answer = initial_response.choices[0].message.content.strip()
    
    # Generate related keywords from the initial answer
    answer_keywords = generate_related_keywords(initial_answer)
    
    # Combine and deduplicate keywords
    all_keywords = list(set(query_keywords + answer_keywords))
    
    # Query the database using the combined keywords
    related_doc = query_db_for_keywords(all_keywords)
    
    # Generate a final response incorporating the related document information
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are College Buddy, an advanced AI assistant designed to help students with their academic queries. Your task is to provide a comprehensive answer based on the initial response and the related document information."""},
            {"role": "user", "content": f"Initial Answer: {initial_answer}\n\nRelated Document: {related_doc}\n\nPlease provide a final answer that incorporates information from the related document, if relevant."}
        ]
    )
   
    final_answer = final_response.choices[0].message.content.strip()
    
    return final_answer, related_doc, all_keywords

# Streamlit Interface
st.set_page_config(page_title="College Buddy Assistant", layout="wide")
st.title("College Buddy Assistant")
st.markdown("Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.")

# Initialize database connection
conn = get_database_connection()
init_db(conn)

# Sidebar for file upload and metadata
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload the Word Documents (DOCX)", type="docx", accept_multiple_files=True)
    if uploaded_files:
        total_token_count = 0
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            text = extract_text_from_docx(uploaded_file)
            token_count = num_tokens_from_string(text)
            total_token_count += token_count
            # Upsert to Pinecone
            upsert_to_pinecone(text, uploaded_file.name, file_id)
            st.text(f"Uploaded: {uploaded_file.name}")
            st.text(f"File ID: {file_id}")
        st.subheader("Uploaded Documents")
        st.text(f"Total token count: {total_token_count}")

# Main content area
st.header("Popular Questions")
# Initialize selected questions in session state
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)

# Display popular questions
for question in st.session_state.selected_questions:
    if st.button(question, key=question):
        with st.spinner("Searching for the best answer..."):
            answer, related_doc, keywords = get_answer(question)
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Related Keywords:")
            st.write(", ".join(keywords))
            
            st.subheader("Related Document:")
            if related_doc:
                with st.expander(f"Document: {related_doc[1]}"):
                    st.write(f"ID: {related_doc[0]}")
                    st.write(f"Title: {related_doc[1]}")
                    st.write(f"Tags: {related_doc[2]}")
                    st.write(f"Link: {related_doc[3]}")
                    
                    # Highlight matching keywords in tags
                    highlighted_tags = related_doc[2]
                    for keyword in keywords:
                        highlighted_tags = highlighted_tags.replace(keyword, f"**{keyword}**")
                    st.markdown(f"Matched Tags: {highlighted_tags}")
            else:
                st.write("No related document found.")
        # Add to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((question, answer))

st.header("Ask Your Own Question")
user_query = st.text_input("What would you like to know about the uploaded documents?")
if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching for the best answer..."):
            answer, related_doc, keywords = get_answer(user_query)
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Related Keywords:")
            st.write(", ".join(keywords))
            
            st.subheader("Related Document:")
            if related_doc:
                with st.expander(f"Document: {related_doc[1]}"):
                    st.write(f"ID: {related_doc[0]}")
                    st.write(f"Title: {related_doc[1]}")
                    st.write(f"Tags: {related_doc[2]}")
                    st.write(f"Link: {related_doc[3]}")
                    
                    # Highlight matching keywords in tags
                    highlighted_tags = related_doc[2]
                    for keyword in keywords:
                        highlighted_tags = highlighted_tags.replace(keyword, f"**{keyword}**")
                    st.markdown(f"Matched Tags: {highlighted_tags}")
            else:
                st.write("No related document found.")
        # Add to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((user_query, answer))
    else:
        st.warning("Please enter a question before searching.")

# Add a section for displaying recent questions and answers
if 'chat_history' in st.session_state and st.session_state.chat_history:
    st.header("Recent Questions and Answers")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.expander(f"Q: {q}"):
            st.write(f"A: {a}")

# Add a section to display the database contents
st.header("Database Contents")
if st.button("Show Database"):
    documents = get_all_documents()
    if documents:
        df = pd.DataFrame(documents, columns=['ID', 'Title', 'Tags', 'Links'])
        st.dataframe(df)
    else:
        st.write("The database is empty.")
