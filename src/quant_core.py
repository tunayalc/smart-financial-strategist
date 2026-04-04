import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
import warnings
import traceback

warnings.filterwarnings('ignore')

TICKER_LIST = ["XU100.IS", "^GSPC", "^IXIC", "GC=F", "SI=F", "BTC-USD", "ETH-USD"]

def get_market_data(tickers, period="10y"):
    print(f"{period} verileri çekiliyor...")
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        
        data = data.dropna()
        
        if data.empty:
            print("Veri yok.")
            return None
            
        print(f"Veri çekildi. Shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        print(traceback.format_exc())
        return None

def get_optimal_portfolio(risk_profile: str, price_data: pd.DataFrame):
    print(f"Risk profili: {risk_profile}")

    try:
        if price_data is None or price_data.empty:
            print("Geçersiz veri")
            return None, None
            
        price_data = price_data.dropna()
        if len(price_data) < 252:  # 1 yıldan az veri
            print("Yetersiz veri")
            return None, None

        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0.01)

        if risk_profile == "düşük":
            print("Hedef: Sharpe maks.")
            ef.max_sharpe()
        elif risk_profile == "orta":
            print("Hedef: %25 volatilite")
            ef.efficient_risk(target_volatility=0.25)
        elif risk_profile == "yüksek":
            print("Hedef: %40 volatilite")
            ef.efficient_risk(target_volatility=0.40)
        else:
            print("Geçersiz profil, varsayılan kullanıldı.")
            ef.max_sharpe()
            
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        print("Optimizasyon tamamlandı.")
        return cleaned_weights, performance

    except OptimizationError as e:
        print(f"Optimizasyon hatası: {e}")
        return None, None
    except Exception as e:
        print(f"Beklenmedik hata: {e}")
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    market_data = get_market_data(TICKER_LIST, period="5y")
    if market_data is not None:
        for risk in ["düşük", "orta", "yüksek"]:
            print("="*50)
            weights, perf = get_optimal_portfolio(risk, market_data)
            if weights:
                print(f"{risk} riskli portföy:")
                print({k: f"{v*100:.2f}%" for k, v in weights.items() if v > 0})
            print("="*50)