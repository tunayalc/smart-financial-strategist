import os

import ollama
import yfinance as yf


def test_market_data() -> None:
    ticker = os.getenv("YFINANCE_TICKER", "XU100.IS")
    print(f"Market data test ({ticker})...")
    try:
        data = yf.download(ticker, period="5d")
        if not data.empty:
            print("Data fetched:")
            print(data.tail(2))
        else:
            print("No data returned.")
    except Exception as exc:
        print(f"Error: {exc}")


def test_ollama_connection() -> None:
    host = os.getenv("OLLAMA_HOST", "http://ollama_core:11434")
    print(f"Ollama connection test ({host})...")
    try:
        client = ollama.Client(host=host)
        response = client.chat(
            model="llama3",
            messages=[{"role": "user", "content": "Hello! Integration test."}],
        )
        print("Ollama responded.")
        print(f"Response: {response['message']['content']}")
    except Exception as exc:
        print(f"Error: {exc}")


def main() -> None:
    print("Running integration tests...")
    test_market_data()
    test_ollama_connection()
    print("Done.")


if __name__ == "__main__":
    main()