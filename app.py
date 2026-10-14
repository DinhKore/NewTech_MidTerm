import streamlit as st
import os
from dotenv import load_dotenv

# Đọc file PDF và cắt nhỏ văn bản
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google Gemini & Vector Database
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Thư viện Prompt & Xử lý đầu ra (LCEL chuẩn mới)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Đọc API Key từ file .env
load_dotenv()

st.set_page_config(page_title="RAG Chatbot Exam", layout="centered")
st.title(" Trợ Lý AI Hỏi Đáp Tài Liệu (RAG)")
st.caption("Midterm Exam: Xây dựng hệ thống RAG tự động nạp Knowledge Base (KB) sử dụng Google Gemini (hoặc là Groq nếu Gemini hết token)")

# Kiểm tra API Key
api_key = os.getenv("GOOGLE_API_KEY")  
if not api_key:
    st.error("Chưa tìm thấy GOOGLE_API_KEY trong file .env! Hãy kiểm tra lại.")
    st.stop()

#api_key_groq = os.getenv("GROQ_API_KEY")
#if not api_key_groq:
#   st.stop()
#   st.warning("Chưa tìm thấy GROQ_API_KEY trong file .env! Nếu bạn muốn dùng Groq, hãy kiểm tra lại.")

# Khởi tạo bộ nhớ tạm 
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

#if "rag_chain" not in st.session_state:
#    st.session_state.rag_chain = None

# --- BƯỚC 2: ĐỌC KB CÓ SẴN VÀ TẠO VECTOR STORE ---
# Tên file Knowledge Base của bạn (Phải đặt cùng thư mục với app.py)
kb_file_path = "KB-B_Sunrise-Family-Clinic-Patient-Guide.md"

# Chỉ đọc và nạp file NẾU hệ thống chưa có dữ liệu trong Session State
if st.session_state.qa_chain is None:
    # Kiểm tra xem file có thực sự nằm ở đó không
    if not os.path.exists(kb_file_path):
        st.error(f"⚠️ Không tìm thấy file `{kb_file_path}`! Hãy copy file này vào cùng thư mục với app.py.")
        st.stop()
        
    with st.spinner("Đang nạp hệ thống Knowledge Base, vui lòng chờ..."):
        # Đọc file markdown bằng TextLoader (nhớ thêm encoding utf-8)
        loader = TextLoader(kb_file_path, encoding='utf-8')
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(raw_documents)

        #Embedding & lưu vào Chroma
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings)
        
        # Lưu retriever vào bộ nhớ tạm
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- BƯỚC 3: MÔ HÌNH LLM & PROMPT ---
    # Đã bổ sung google_api_key để tránh lỗi xác thực
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=api_key)

    ##Dự phòng Groq
    #from langchain_groq import ChatGroq
    #llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))

    prompt_template = """Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp dưới đây.
Nếu không có thông tin trong ngữ cảnh, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu'.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Ghép chuỗi RAG hiện đại bằng LCEL (|) và lưu vào Session State
    st.session_state.qa_chain = prompt | llm | StrOutputParser()

    st.success("Tài liệu đã được nạp vào AI!")

# --- BƯỚC 4: HỎI ĐÁP & HIỂN THỊ NGUỒN ---
if "qa_chain" in st.session_state and st.session_state.qa_chain is not None:
    user_query = st.text_input("Nhập câu hỏi bạn muốn hỏi về tài liệu:")
    
    if user_query:
        with st.spinner("AI đang tìm kiếm thông tin và tạo câu trả lời..."):
            # 1. Tự tay gọi Retriever để lấy các đoạn văn bản (Chunks) & Metadata
            retrieved_docs = st.session_state.retriever.invoke(user_query)
            
            # 2. Gộp nội dung chữ để đưa cho AI
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # 3. Yêu cầu AI trả lời dựa trên context_text vừa tạo
            answer = st.session_state.qa_chain.invoke({
                "context": context_text, 
                "question": user_query
            })
            
            # 4. In kết quả trả lời của AI
            st.markdown("### Kết quả trả lời:")
            st.write(answer)
            
            # 5. IN NGUỒN VÀ RETRIEVED CHUNKS 
            st.markdown("---")
            # Dùng st.expander tạo khung sổ xuống (rất chuyên nghiệp)
            with st.expander(" Xem nguồn trích dẫn & Văn bản gốc (Retrieved Chunks)"):
                for i, doc in enumerate(retrieved_docs):
                    # Lấy số trang và tên file từ metadata 
                    page_num = doc.metadata.get('page', 0) + 1 
                    source_file = doc.metadata.get('source', 'Unknown')
                    
                    st.markdown(f"**Trích dẫn {i+1}:** *Từ file `{source_file}` - Trang {page_num}*")
                    st.info(doc.page_content) # In ra đoạn chunk (văn bản thô)