import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

urls = [
    'https://www.pdx.edu/engineering/',
    'https://www.pdx.edu/engineering/departments-programs',
    'https://www.pdx.edu/computer-science/',
    'https://www.pdx.edu/computer-science/graduate',
    'https://www.pdx.edu/computer-science/master',
    'https://www.pdx.edu/computer-science/masters-track-courses',
    'https://docs.google.com/spreadsheets/d/1Zzyb9E1zLwQ0TYErZfoW9i2BM83b_PFba6zWmzMELQs/edit#gid=0',
    'https://www.pdx.edu/engineering/academic-programs',
    'https://www.pdx.edu/gradschool/graduate-candidate-deadlines'
]

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

def get_vectorstore(url):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 700,
                        chunk_overlap = 200
                    )
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002"
                    )
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store
    
def get_context_retriever_chain(vector_store):

    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(template)

    primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    retrieval_augmented_qa_chain = (
    
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
   
    | RunnablePassthrough.assign(context=itemgetter("context"))
    
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    return retrieval_augmented_qa_chain

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    response = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "question": user_query
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="PSU Buddy", page_icon="🤖")
st.title("PSU Buddy")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore(urls)    

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)