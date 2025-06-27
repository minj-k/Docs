import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time

# 파일 처리 라이브러리
import PyPDF2
from docx import Document
import pyhwp

# --- 초기 설정 ---
st.set_page_config(page_title="[1단계] RAG 공문서 생성기", page_icon="✍️")
load_dotenv()

# --- API 키 및 모델 설정 ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # 모델 설정
    embedding_model = "models/text-embedding-004"
    generation_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("API 키를 설정하는 중 오류가 발생했습니다. .env 파일을 확인해주세요.")
    st.stop()


# --- RAG 핵심 기능: 1. 데이터베이스 구축 ---
@st.cache_resource(show_spinner="AI 도서관 구축 중... (앱 최초 실행 시 몇 분 소요될 수 있습니다)")
def build_rag_database(data_path):
    """'data' 폴더의 모든 문서를 읽고, 텍스트를 추출하고, AI로 임베딩하여 검색 가능한 데이터베이스를 만듭니다."""
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.hwp', '.pdf', '.docx', '.txt'))]
    
    if not all_files:
        return None

    documents = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        text = ""
        try:
            if file_name.lower().endswith('.hwp'):
                with open(file_path, 'rb') as f:
                    hwp = pyhwp.HWPReader(f)
                    text = hwp.get_text()
            elif file_name.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() if page.extract_text() else ""
            elif file_name.lower().endswith('.docx'):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + '\n'
            elif file_name.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if text.strip():
                documents.append({'filename': file_name, 'text': text})
        except Exception as e:
            st.warning(f"'{file_name}' 처리 중 오류 발생: {e}", icon="⚠️")

    if not documents:
        return None

    df = pd.DataFrame(documents)
    
    def embed_fn(texts):
        # 여러 텍스트를 한 번에 처리하여 효율성 증대
        return genai.embed_content(model=embedding_model, content=texts, task_type="RETRIEVAL_DOCUMENT")["embedding"]

    # 모든 텍스트를 한 번에 임베딩
    st.info(f"{len(df)}개 문서의 의미를 분석하여 'AI 도서관'에 등록하고 있습니다...")
    df['embeddings'] = embed_fn(df['text'].tolist())
    
    return df

# --- RAG 핵심 기능: 2. 관련 문서 검색 ---
def find_best_passage(query, dataframe):
    """사용자 쿼리와 가장 유사한 단일 문서를 데이터베이스에서 찾습니다."""
    query_embedding = genai.embed_content(model=embedding_model, content=query, task_type="RETRIEVAL_QUERY")["embedding"]
    dot_products = np.dot(np.stack(dataframe['embeddings']), query_embedding)
    index = np.argmax(dot_products)
    return dataframe.iloc[index]['text']

# --- AI 생성 함수 ---
def generate_gongmun_with_rag(user_request, reference_passage):
    """검색된 참고자료와 사용자 요청을 바탕으로 AI가 최종 공문을 생성합니다."""
    prompt = f"""
    당신은 대한민국 공공기관의 유능한 행정 전문가입니다. 사용자의 '새로운 요청'에 맞춰 공문서 초안을 작성해야 합니다.
    작성을 돕기 위해, 과거에 작성했던 가장 유사한 '참고 자료'를 제공합니다.
    '참고 자료'의 전체적인 형식, 스타일, 문체, 주요 내용을 적극적으로 참고하여, 사용자의 '새로운 요청'에 맞는 완벽한 공문서 초안을 작성해주세요.

    --- 참고 자료 (과거 베스트 프랙티스) ---
    {reference_passage}

    --- 사용자의 새로운 요청 ---
    {user_request}

    --- 공문서 초안 작성 시작 ---
    """
    
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 생성 중 오류 발생: {e}"

# --- Streamlit UI 구성 ---
st.title("✍️ [1단계] RAG 기반 공문서 생성기")
st.markdown("---")

# 1. 데이터베이스 구축
data_folder = "data"
if not os.path.exists(data_folder):
    st.error(f"'{data_folder}' 폴더를 찾을 수 없습니다. 프로젝트 폴더 내에 'data' 폴더를 만들고 문서 파일들을 넣어주세요.")
    st.stop()

db = build_rag_database(data_folder)

if db is not None:
    st.success(f"✅ AI 도서관 구축 완료! 총 {len(db)}개의 문서가 등록되었습니다.")
    st.markdown("---")
    
    st.info("아래에 간단한 지시사항만 입력하면, AI가 '도서관'에서 최적의 참고자료를 찾아 공문을 생성합니다.")

    # 2. 사용자 요청 입력
    request_placeholder = "예시: 다음주 수요일(7월 2일) 오후 2시에 있을 'AI 기술고도화 사업 현장점검' 계획보고서 초안 작성해줘. 대상은 (주)에코팜이고, 참석자는 나(이채민)랑 강병범 선임이야."
    user_request = st.text_area(
        "어떤 공문서를 생성할까요?",
        height=150,
        placeholder=request_placeholder
    )

    # 3. 공문 생성 버튼
    if st.button("🚀 공문 생성 시작", type="primary"):
        if not user_request:
            st.warning("요청사항을 입력해주세요.")
        else:
            # RAG 검색 및 생성 실행
            with st.spinner("AI 사서가 도서관에서 가장 관련있는 문서를 찾고 있습니다..."):
                time.sleep(1) # 시각적 효과를 위해 잠시 대기
                reference_doc = find_best_passage(user_request, db)
                
            st.success(f"가장 유사한 과거 문서를 참고자료로 찾았습니다!")

            with st.spinner("찾아낸 참고자료를 바탕으로 공문서 초안을 작성 중입니다..."):
                time.sleep(1)
                generated_gongmun = generate_gongmun_with_rag(user_request, reference_doc)
                
            # 결과 표시
            st.markdown("---")
            st.subheader("🎉 AI가 작성한 공문서 초안")
            st.text_area("결과", generated_gongmun, height=600)
            st.download_button("결과를 텍스트 파일로 다운로드", generated_gongmun, "generated_gongmun.txt")
else:
    st.warning("'data' 폴더에 분석할 문서가 없습니다. HWP, PDF, DOCX, TXT 파일을 넣어주세요.")
