from urllib.parse import urljoin
import requests
import time
import pandas as pd

# base URL for GET request to obtain trade data
BASEURL = 'https://api.bitfinex.com/v2/trades/'
# coin pair to obtain trades for
SYMBOL = 'tBTCUSD'
# number of trades to obtain per query
LIMIT = 1000
# number of total requests to send
REQUESTS = 10
# time between requests in seconds
WAIT = 1

# default value -> get the latest trades
last_timestamp = 0
successful_requests = 0
trades = []
for _ in range(REQUESTS):
    url = urljoin(BASEURL, f'{SYMBOL}/hist')
    query_time = time.time()
    # get <LIMIT> trades since last timestamp
    print(url)
    response = requests.get(url,
                            params={'limit': LIMIT,
                                    'start': last_timestamp})
    # if the request was successful, update data
    if response.status_code == 200:
        print('Request successful')
        successful_requests += 1
        trade_data = pd.DataFrame(response.json(),
                                  columns=['ID', 'MTS', 'AMOUNT', 'PRICE'])
        print(f'{len(trade_data)} trades since last timestamp')
        last_timestamp = max(trade_data['MTS'])
        trades.append(trade_data)
        # make sure we adhere to <WAIT> period
        timedelta = time.time() - query_time
        if timedelta < WAIT:
            time.sleep(WAIT - timedelta)

print(f'Completed {successful_requests} requests successfully')

all_trades = pd.concat(trades)
all_trades.to_parquet(f'{SYMBOL}_trades.parquet')
