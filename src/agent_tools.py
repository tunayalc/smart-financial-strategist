from langchain_core.tools import tool
from typing import Dict, Any
import traceback

try:
    from quant_core import get_market_data, get_optimal_portfolio, TICKER_LIST
    from rag_core import build_vector_database, query_rag
except ImportError as e:
    print(f"Import hatası: {e}")
    print(traceback.format_exc())
    raise

@tool
def get_portfolio_tool(risk_profile: str) -> Dict[str, Any]:
    """
    Kullanıcının risk profiline göre optimal portföy dağılımı ve performans metriklerini hesaplar.
    
    Args:
        risk_profile (str): Risk profili - 'düşük', 'orta', veya 'yüksek'
        
    Returns:
        Dict: Portföy ağırlıkları ve performans metrikleri
    """
    try:
        print(f"get_portfolio_tool çalıştırıldı (risk_profile='{risk_profile}')")
        
        market_data = get_market_data(TICKER_LIST, period="5y")
        if market_data is None or market_data.empty:
            return {"error": "Piyasa verileri alınamadı."}
        
        weights, performance = get_optimal_portfolio(risk_profile, market_data)
        if weights is None:
            return {"error": "Optimizasyon hatası."}

        non_zero_weights = {k: v for k, v in weights.items() if v > 0.01}
        
        return {
            "weights": non_zero_weights,
            "performance": {
                "expected_return": round(performance[0], 4),
                "annual_volatility": round(performance[1], 4),
                "sharpe_ratio": round(performance[2], 4)
            }
        }
    except Exception as e:
        error_msg = f"Portfolio tool hatası: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}

try:
    RAG_DATABASE = build_vector_database()
except Exception as e:
    print(f"RAG database oluşturma hatası: {e}")
    RAG_DATABASE = None

@tool
def get_justification_tool(topic: str) -> str:
    """
    Belirtilen konu hakkında bilgi tabanından ilgili açıklama ve gerekçeleri getirir.
    
    Args:
        topic (str): Hakkında bilgi istenilen konu (örn: "altın", "çeşitlendirme")
        
    Returns:
        str: Konuyla ilgili açıklayıcı metin
    """
    try:
        print(f"get_justification_tool çalıştırıldı (topic='{topic}')")
        
        if RAG_DATABASE is None:
            return "Bilgi veritabanı yüklenemedi."
        
        results = query_rag(RAG_DATABASE, query=f"{topic} hakkında bilgi ver", k=1)
        if not results:
            return f"'{topic}' için bilgi yok."
            
        return results[0].page_content
    except Exception as e:
        error_msg = f"Justification tool hatası: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg