import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

from src.config import settings
from src.graph import ChatbotGraphBuilder

# Page configuration
st.set_page_config(
    page_title="Agentic Web Chatbot",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stChatMessage { border-radius: 12px; padding: 12px; margin-bottom: 10px; }
    .stButton>button { border-radius: 8px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

st.title("Agentic Web Search Chatbot")
st.caption("Architecture: LangGraph StateGraph, MemorySaver Checkpointers, HITL Tool Approval, and Telemetry")

# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    
    api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        
    tavily_key_input = st.text_input("Tavily API Key (Optional)", type="password", value=os.getenv("TAVILY_API_KEY", ""))
    if tavily_key_input:
        os.environ["TAVILY_API_KEY"] = tavily_key_input

    model_name = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.divider()
    st.subheader("LangGraph Configuration")
    
    enable_web_search = st.toggle("Enable Web Search Tools", value=True)
    enable_hitl = st.toggle("Human-In-The-Loop (HITL) Tool Interrupt", value=False, help="Requires explicit human approval before running web search tools.")
    
    st.info(f"Active Session Thread ID: {st.session_state.thread_id}")

    if st.button("New Thread Session"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.status("Tool Execution Output", expanded=False):
            st.write(msg.content)

# User Chat Input
if prompt := st.chat_input("Ask me anything..."):
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a valid Groq API Key in the sidebar.")
        st.stop()

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    try:
        settings.configure_langsmith()
        
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.7
        )
        
        builder = ChatbotGraphBuilder(llm=llm)
        graph = builder.build_agentic_search_graph(hitl_interrupt=enable_hitl) if enable_web_search else builder.build_basic_graph()

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.chat_message("assistant"):
            with st.spinner("Executing StateGraph transition..."):
                response_state = graph.invoke({"messages": st.session_state.messages}, config=config)
                
                new_messages = response_state["messages"][len(st.session_state.messages):]
                
                for msg in new_messages:
                    if isinstance(msg, AIMessage) and msg.content:
                        st.write(msg.content)
                    elif isinstance(msg, ToolMessage):
                        with st.status("Tool Result", expanded=False):
                            st.write(msg.content)
                            
                st.session_state.messages = response_state["messages"]

    except Exception as e:
        st.error(f"Error executing agentic workflow: {e}")
