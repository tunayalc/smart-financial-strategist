# smart-financial-strategist

N Kolay Ödeme Kuruluşu A.Ş. stajım sırasında geliştirdiğim AI destekli finansal strateji prototipi

## Genel Bakış

`smart-financial-strategist`, nicel finans hesaplamalarını, yerel bilgi tabanını ve LLM tabanlı açıklama üretimini tek akışta birleştiren bir proje. Sistem, yalnızca serbest metin yanıtı üretmek yerine, araç kullanan ve dayanak üreten bir finansal öneri katmanı kurmayı hedefliyor.

## Sistem Katmanları

| Katman | İçerik |
| --- | --- |
| Arayüz | Streamlit tabanlı kullanıcı deneyimi |
| Agent Orkestrasyonu | LangChain tabanlı tool-calling yaklaşımı |
| Quant Layer | piyasa verisi çekme ve portföy optimizasyonu |
| RAG Layer | yerel bilgi tabanından açıklama getirme |
| LLM Layer | nihai öneriyi doğal dilde üretme |

## Ana Bileşenler

### `src/app.py`

Kullanıcıdan gelen hedefi alıp tüm araç çağrılarını tetikleyen ana arayüz katmanı.

### `src/main_agent.py`

Arayüz dışında ajan mantığını test etmeye yarayan script.

### `src/agent_tools.py`

LLM ajanının quant ve RAG bileşenlerine erişmesini sağlayan araç köprüsü.

### `src/quant_core.py`

Piyasa verisi üzerinden portföy önerisi üreten nicel hesaplama katmanı.

### `src/rag_core.py`

Yerel finans metinlerinden embedding üretip ilgili içerikleri geri getiren açıklama katmanı.

### `src/knowledge_base/`

Finansal gerekçelendirme üretiminde kullanılan yerel bilgi tabanı.

## Repo Yapısı

```text
smart-financial-strategist/
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- scripts/
|   `-- integration_test.py
|-- src/
|   |-- agent_tools.py
|   |-- app.py
|   |-- main_agent.py
|   |-- quant_core.py
|   |-- rag_core.py
|   `-- knowledge_base/
`-- README.md
```

## Kullanılan Teknolojiler

- Python
- Streamlit
- LangChain
- Ollama
- Chroma
- sentence-transformers
- yfinance
- PyPortfolioOpt
