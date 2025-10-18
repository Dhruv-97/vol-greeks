import os, requests, json, sys, csv
from datetime import datetime
import requests

API_KEY = os.environ.get("MARKETDATA_API_KEY")


# symbol = "SPY"


# params = {
#     "symbol": symbol,
#     "token": API_KEY,
# }


def get(symbol, token):
    BASE_URL_2 = "https://api.marketdata.app/v1/options/chain/{symbol}"

    response = requests.get(BASE_URL_2.format(symbol=symbol), params={"token": token}, timeout=20)
    print("REQUEST URL:", response.url)
    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        response.raise_for_status()
    return response.json()

if __name__ == "__main__":
        
    try:
        data = get('TSLA', API_KEY)
    except requests.HTTPError as e:
        print("HTTP Error:", e)

    if data['s'] == 'ok':
        # print(f"Fetched options chain for {symbol} successfully.")
        for i in range(len(data['optionSymbol'])):
            print(f"{data['optionSymbol'][i]} {data['strike'][i]} {data['side'][i]} {data['dte'][i]} @ {data['mid'][i]}")
    else:
        print(f"Error fetching options chain for {symbol}: {data.get('errmsg', 'Unknown error')}")
        sys.exit(1)

