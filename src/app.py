import streamlit as st
import traceback
import sys
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agent_tools import get_portfolio_tool, get_justification_tool

st.set_page_config(
    page_title="Akıllı Finansal Stratejist",
    layout="wide"
)

st.title("Akıllı Finansal Stratejist")
st.caption("Yapay zeka destekli kişiye özel yatırım yol haritanız.")

@st.cache_resource
def get_agent_executor():
    try:
        print("Ajan ve araçlar kuruluyor...")
        
        llm = ChatOllama(
            model="llama3.1", # <-- DÜZELTİLDİ
            base_url="http://ollama_core:11434",
            temperature=0.1
        )
        
        tools = [get_portfolio_tool, get_justification_tool]
        
        prompt_template = """Sen bir finansal stratejistsin.
Görevin, kullanıcının hedeflerini analiz ederek ona özel, gerekçelendirilmiş ve anlaşılır bir yatırım yol haritası oluşturmaktır.

Süreç:
1. Kullanıcının girdisinden risk profilini belirle ('düşük', 'orta', 'yüksek')
2. get_portfolio_tool ile portföy dağılımını ve performansı hesapla
3. Her varlık için get_justification_tool ile gerekçe oluştur
4. Sonuçları Markdown formatında raporla

Kullanıcının Sorusu: {input}
Ara Adımlar: {agent_scratchpad}
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    except Exception as e:
        print(f"Agent oluşturma hatası: {e}")
        print(traceback.format_exc())
        raise e

def test_ollama_connection():
    try:
        llm = ChatOllama(
            model="llama3.1", # <-- DÜZELTİLDİ
            base_url="http://ollama_core:11434"
        )
        response = llm.invoke("Test")
        return True, "Bağlantı başarılı (llama3.1)"
    except Exception as e:
        return False, str(e)

st.sidebar.header("Hedefiniz")
user_goal = st.sidebar.text_area(
    "Finansal hedefinizi yazın:", 
    "10 yıl içinde emeklilik için birikim yapmak istiyorum. Orta düzeyde risk alabilirim.",
    height=150
)

if st.sidebar.button("Bağlantıyı Test Et"):
    with st.spinner("Ollama bağlantısı test ediliyor..."):
        success, message = test_ollama_connection()
        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(f"Bağlantı hatası: {message}")

if st.sidebar.button("Yol Haritamı Oluştur"):
    if user_goal:
        st.subheader("Finansal Yol Haritanız")
        progress_bar = st.progress(0, text="Hazırlanıyor...")
        
        try:
            progress_bar.progress(25, text="Agent yükleniyor...")
            agent_executor = get_agent_executor()
            
            progress_bar.progress(50, text="Hesaplanıyor...")
            
            with st.spinner("Portföy optimizasyonu yapılıyor..."):
                result = agent_executor.invoke({"input": user_goal})
            
            progress_bar.progress(100, text="Rapor hazır")
            
            if result and "output" in result:
                st.markdown(result["output"])
            else:
                st.error("Agent sonuç üretemedi")
                
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            st.error(f"Hata oluştu: {error_msg}")
            
            with st.expander("Teknik Detaylar"):
                st.code(error_trace)
            
            print(f"Streamlit hatası: {error_msg}")
            print(f"Stack trace: {error_trace}")
            
            progress_bar.empty()
    else:
        st.sidebar.error("Lütfen bir hedef girin.")
else:
    st.info("Sol menüden hedefinizi yazıp butona basın.")