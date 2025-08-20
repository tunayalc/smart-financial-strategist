import yfinance as yf
import ollama
import pandas as pd

def test_veri_cekme():
    print("Yahoo Finance testi...")
    try:
        data = yf.download("XU100.IS", period="5d")
        if not data.empty:
            print("BIST 100 verisi çekildi.")
            print(data.tail(2))
        else:
            print("Veri çekilemedi.")
    except Exception as e:
        print(f"Hata: {e}")

def test_ollama_baglantisi():
    print("Ollama bağlantı testi...")
    try:
        client = ollama.Client(host='http://ollama_core:11434')
        response = client.chat(model='llama3', messages=[
          {
            'role': 'user',
            'content': 'Merhaba! Bağlantı testi yapıyorum.',
          },
        ])
        print("Ollama cevap verdi.")
        print(f"Cevap: {response['message']['content']}")
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    print("Entegrasyon testleri başlıyor...")
    test_veri_cekme()
    test_ollama_baglantisi()
    print("Testler tamamlandı.")
