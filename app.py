# Kaito.ai Chatbot - Streamlit Deployment
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Tuple
import re
import time
from datetime import datetime

# ===============================
# PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="Kaito.ai Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS STYLING
# ===============================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }

    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }

    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }

    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .sidebar-logo {
        display: block;
        margin: 0 auto 2rem auto;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# DATA CLASSES (Same as before)
# ===============================

class KaitoDataProcessor:
    """Process and prepare Kaito.ai data for the chatbot"""

    def __init__(self):
        self.data = []
        self.processed_chunks = []

    def add_text_data(self, text: str, source: str = "manual"):
        """Add text data about Kaito.ai"""
        chunks = self.chunk_text(text, max_length=500)
        for i, chunk in enumerate(chunks):
            self.data.append({
                'content': chunk,
                'source': source,
                'chunk_id': f"{source}_{i}"
            })

    def add_qa_pairs(self, qa_pairs: List[Dict]):
        """Add question-answer pairs"""
        for qa in qa_pairs:
            self.data.append({
                'content': f"Q: {qa['question']} A: {qa['answer']}",
                'source': 'qa_pair',
                'question': qa['question'],
                'answer': qa['answer']
            })

    def chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into smaller chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_processed_data(self) -> List[Dict]:
        """Return processed data"""
        return self.data

class KaitoAIChatbot:
    """Main chatbot class for Kaito.ai"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.data = []
        self.embeddings = None
        self.index = None
        self.processor = KaitoDataProcessor()

    def load_data(self, data: List[Dict]):
        """Load processed data into the chatbot"""
        self.data = data

    @st.cache_resource
    def create_embeddings(_self):
        """Create embeddings for all data"""
        contents = [item['content'] for item in _self.data]
        _self.embeddings = _self.model.encode(contents)

        # Create FAISS index for fast similarity search
        dimension = _self.embeddings.shape[1]
        _self.index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(_self.embeddings)
        _self.index.add(_self.embeddings)

        return True

    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar content"""
        if self.index is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append((self.data[idx], float(score)))

        return results

    def generate_answer(self, query: str, max_context_length: int = 1000) -> str:
        """Generate answer based on retrieved context"""
        similar_docs = self.search_similar(query, top_k=3)

        if not similar_docs or similar_docs[0][1] < 0.3:
            return "I don't have enough information about that specific aspect of Kaito.ai. Could you rephrase your question or ask about something else related to Kaito.ai?"

        # Combine top results for context
        context = ""
        for doc, score in similar_docs:
            if len(context) + len(doc['content']) < max_context_length:
                context += doc['content'] + "\n\n"

        # Simple answer extraction
        answer = self.extract_answer(query, context)
        return answer

    def extract_answer(self, query: str, context: str) -> str:
        """Extract answer from context"""
        context_lower = context.lower()
        query_lower = query.lower()

        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)

        # Find most relevant sentences
        relevant_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in query_lower.split()):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        else:
            return context[:500] + "..." if len(context) > 500 else context

    def chat(self, query: str) -> str:
        """Main chat function"""
        return self.generate_answer(query)

# ===============================
# SAMPLE DATA (Your 200 Q&A pairs)
# ===============================

@st.cache_data
def load_kaito_data():
    """Load the comprehensive Kaito.ai dataset"""
    # Include the full 200 Q&A pairs here (truncated for brevity)
    sample_kaito_data = [
        {
            'question': 'What is Kaito.ai?',
            'answer': 'Kaito.ai is an AI-powered platform that serves as the distribution center for crypto, facilitating the seamless flow of information, attention, and capital through advanced artificial intelligence technology.'
        },
        {
            'question': 'Who founded Kaito.ai?',
            'answer': 'Kaito.ai was founded in 2022 by Yu Hu, a Cambridge University graduate and former Citadel hedge fund portfolio manager.'
        },
        {
            'question': 'What is the main problem Kaito.ai solves?',
            'answer': 'Kaito.ai solves the problem of information fragmentation in the cryptocurrency space, where critical data is scattered across countless platforms making it nearly impossible for users to comprehensively grasp industry developments.'
        },
        {
            'question': 'What is InfoFi?',
            'answer': 'InfoFi (Information Finance) is a concept where information flows are driven by market forces rather than centralized algorithms, creating a more efficient and fair system that rewards participants appropriately.'
        },
        {
            'question': 'How much funding has Kaito.ai raised?',
            'answer': 'Kaito.ai has raised $10.8 million across two funding rounds, with notable investors including Dragonfly Capital, Sequoia Capital China, Superscrypt, and Spartan Group.'
        },
        {
            'question': 'What is Kaito Pro?',
            'answer': 'Kaito Pro is an AI-powered vertical search engine specifically designed for the cryptocurrency industry that indexes thousands of Web3 sources to provide real-time, high-quality market intelligence.'
        },
        {
            'question': 'What are Kaito Yaps?',
            'answer': 'Kaito Yaps are a tokenized attention mechanism that uses AI to quantify and distribute attention based on relevance and impact, representing engagement points within the Kaito ecosystem.'
        },
        {
            'question': 'How do you earn Yaps?',
            'answer': 'Users earn Yaps by getting verified, creating quality content, engaging thoughtfully with smart accounts, and contributing valuable crypto-related insights on social media platforms like Twitter.'
        },
        {
            'question': 'What is the KAITO token?',
            'answer': 'KAITO is the native cryptocurrency of the Kaito ecosystem, serving as the backbone for governance, transactions, market incentives, and influencing attention distribution within the network.'
        },
        {
            'question': 'What blockchain is KAITO built on?',
            'answer': 'The KAITO token is built on Base Chain, providing a foundation for the InfoFi network with efficient and scalable blockchain infrastructure.'
        }
        # Add more Q&A pairs here...
    ]
    return sample_kaito_data

# ===============================
# STREAMLIT APP INITIALIZATION
# ===============================

@st.cache_resource
def initialize_chatbot():
    """Initialize and train the chatbot"""
    chatbot = KaitoAIChatbot()

    # Load data
    kaito_data = load_kaito_data()
    chatbot.processor.add_qa_pairs(kaito_data)

    # Process data
    processed_data = chatbot.processor.get_processed_data()
    chatbot.load_data(processed_data)

    # Create embeddings
    chatbot.create_embeddings()

    return chatbot

# ===============================
# MAIN APPLICATION
# ===============================

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Kaito.ai Expert Chatbot</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150/667eea/white?text=KAITO",
                caption="Kaito.ai Assistant",
                width=150,
                use_column_width=False)

        st.markdown("### 📊 Chatbot Stats")

        # Stats boxes
        st.markdown("""
        <div class="stats-box">
            <h3>200+</h3>
            <p>Knowledge Base Entries</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stats-box">
            <h3>AI-Powered</h3>
            <p>Semantic Search</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stats-box">
            <h3>Real-time</h3>
            <p>Responses</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Sample questions
        st.markdown("### 💡 Try asking:")
        sample_questions = [
            "What is Kaito.ai?",
            "How do Yaps work?",
            "What is Kaito Pro?",
            "How to earn KAITO tokens?",
            "What is InfoFi?",
            "Who founded Kaito?",
            "What makes Kaito different?",
            "How does the attention economy work?"
        ]

        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.sample_question = question

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This chatbot is powered by AI and trained on comprehensive
        information about Kaito.ai, including their InfoFi network,
        Yaps system, and KAITO token.
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            with st.spinner('🚀 Initializing Kaito.ai Expert Chatbot...'):
                st.session_state.chatbot = initialize_chatbot()
            st.success('✅ Chatbot ready! Ask me anything about Kaito.ai!')

        # Chat interface
        st.markdown("### 💬 Chat with Kaito.ai Expert")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Chat input
        user_input = st.text_input(
            "Ask me anything about Kaito.ai:",
            placeholder="e.g., What is the KAITO token used for?",
            key="user_input"
        )

        # Handle sample question clicks
        if 'sample_question' in st.session_state:
            user_input = st.session_state.sample_question
            del st.session_state.sample_question

        # Process user input
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'type': 'user',
                'message': user_input,
                'timestamp': datetime.now()
            })

            # Generate response
            with st.spinner('🤔 Thinking...'):
                response = st.session_state.chatbot.chat(user_input)

            # Add bot response to history
            st.session_state.chat_history.append({
                'type': 'bot',
                'message': response,
                'timestamp': datetime.now()
            })

        # Display chat history
        st.markdown("### 📝 Conversation History")

        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {chat['message']}
                        <br><small>🕐 {chat['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Kaito Expert:</strong> {chat['message']}
                        <br><small>🕐 {chat['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👋 Start a conversation by asking a question about Kaito.ai!")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        # Features showcase
        st.markdown("### ⚡ Key Features")

        features = [
            {
                "title": "🔍 Comprehensive Knowledge",
                "description": "200+ curated Q&A pairs covering all aspects of Kaito.ai"
            },
            {
                "title": "🧠 AI-Powered Search",
                "description": "Semantic similarity matching for accurate responses"
            },
            {
                "title": "⚡ Real-time Responses",
                "description": "Instant answers to your Kaito.ai questions"
            },
            {
                "title": "📱 User-Friendly",
                "description": "Simple interface designed for easy interaction"
            }
        ]

        for feature in features:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{feature['title']}</h4>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick info
        st.markdown("### 📚 Topics Covered")
        topics = [
            "Kaito Pro Search Engine",
            "Yaps & Attention Economy",
            "KAITO Token & Tokenomics",
            "InfoFi Network",
            "Technology & AI",
            "Community & Ecosystem",
            "Investment & Funding",
            "Use Cases & Benefits"
        ]

        for topic in topics:
            st.markdown(f"• {topic}")

if __name__ == "__main__":
    main()

# ===============================
# REQUIREMENTS.TXT CONTENT
# ===============================

"""
Create a requirements.txt file with these dependencies:

streamlit>=1.28.0
sentence-transformers>=2.2.2
torch>=2.0.0
faiss-cpu>=1.7.4
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
"""