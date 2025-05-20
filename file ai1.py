import requests
import threading
import time
import unicodedata
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class VNStockRecommender:
    """
    Chatbot recommends Vietnamese stocks using VNDirect API and a DL model.
    Graceful fallback if network/API fails.
    """
    def __init__(self, model_path='financial_report_model.h5'):
        self.exchanges = {
            'hose': ['VIC', 'VHM', 'MSN'],
            'hnx': ['SHS', 'PVS'],
            'upcom': ['VND', 'GAS']
        }
        self.quote_url = 'https://finfo-api.vndirect.com.vn/v4/quotes'
        self.fs_url = 'https://finfo-api.vndirect.com.vn/v4/financial-statements'
        self.cache = {}
        self.lock = threading.Lock()
        try:
            self.dl_model = load_model(model_path)
        except Exception:
            self.dl_model = None

    def normalize_text(self, text):
        nk = unicodedata.normalize('NFD', text)
        return ''.join(c for c in nk if unicodedata.category(c) != 'Mn').lower()

    def safe_request(self, url, params, timeout=5):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[WARN] API request failed: {e}")
            return None

    def fetch_quote(self, symbol):
        data = self.safe_request(self.quote_url, {'code': symbol})
        if data and data.get('data'):
            return data['data'][0]
        return {'close': 0.0, 'pe': np.nan, 'pb': np.nan}

    def fetch_price(self, symbol):
        return float(self.fetch_quote(symbol).get('close', 0.0))

    def fetch_pe(self, symbol):
        return float(self.fetch_quote(symbol).get('pe', np.nan))

    def fetch_pb(self, symbol):
        return float(self.fetch_quote(symbol).get('pb', np.nan))

    def fetch_history(self, symbol, days=30):
        price = self.fetch_price(symbol)
        return [price] * days

    def fetch_financial_report(self, symbol):
        data = self.safe_request(self.fs_url, {'code': symbol, 'type': 'AR'}, timeout=10)
        if data and data.get('data') and 'financeStatements' in data['data']:
            fs = data['data']['financeStatements']
            if fs:
                rec = fs[0]
                return pd.DataFrame([{  
                    'current_ratio': float(rec.get('currentRatio', np.nan)),
                    'debt_to_equity': float(rec.get('debtToEquity', np.nan)),
                    'net_profit_margin': float(rec.get('netProfitMargin', np.nan)),
                    'revenue_growth': float(rec.get('revenueGrowth', np.nan))
                }])
        # Fallback dummy data
        return pd.DataFrame([{  
            'current_ratio': 1.0,
            'debt_to_equity': 1.0,
            'net_profit_margin': 0.0,
            'revenue_growth': 0.0
        }])

    def analyze_financials(self, symbol):
        df = self.fetch_financial_report(symbol)
        X = df.values.astype('float32')
        if self.dl_model is None:
            return 0.5
        return float(self.dl_model.predict(X)[0, 0])

    def analyze(self, symbol):
        with self.lock:
            cached = self.cache.get(symbol)
            if cached and time.time() - cached['ts'] < 300:
                return cached['res']
        price = self.fetch_price(symbol)
        pe = self.fetch_pe(symbol)
        pb = self.fetch_pb(symbol)
        hist = self.fetch_history(symbol)
        sma5 = sum(hist[-5:]) / 5
        sma20 = sum(hist[-20:]) / 20
        trend = 'up' if sma5 > sma20 else 'down'
        score = self.analyze_financials(symbol)
        res = {
            'symbol': symbol,
            'price': price,
            'pe': pe,
            'pb': pb,
            'sma5': sma5,
            'sma20': sma20,
            'trend': trend,
            'dl_score': score
        }
        with self.lock:
            self.cache[symbol] = {'res': res, 'ts': time.time()}
        return res

    def recommend_symbol(self, symbol):
        m = self.analyze(symbol)
        if m['pb'] >= 0.9:
            return f"{symbol}: Skip P/B={m['pb']:.2f}"
        if m['trend'] == 'up' and m['pe'] < 15 and m['dl_score'] > 0.6:
            return f"BUY {symbol} | P/E={m['pe']:.1f} | P/B={m['pb']:.2f} | Score={m['dl_score']:.2f}"
        if m['trend'] == 'down' or m['dl_score'] < 0.4:
            return f"SELL {symbol} | Trend={m['trend']} | Score={m['dl_score']:.2f}"
        return f"HOLD {symbol} | Score={m['dl_score']:.2f}"

    def handle(self, text):
        cmd = self.normalize_text(text)
        if 'list' in cmd:
            return 'Available exchanges: ' + ', '.join(self.exchanges.keys())
        if 'recommend' in cmd:
            parts = cmd.split()
            if len(parts) == 2 and parts[1] in self.exchanges:
                syms = self.exchanges[parts[1]]
                return '\n'.join(self.recommend_symbol(s) for s in syms)
            if len(parts) == 2:
                return self.recommend_symbol(parts[1].upper())
            # Default: all exchanges
            out = []
            for ex, syms in self.exchanges.items():
                out.append(f"--{ex.upper()}--")
                out += [self.recommend_symbol(s) for s in syms]
            return '\n'.join(out)
        return "Enter 'list' or 'recommend <exchange|symbol>'."

# Training function

def train_financial_model(symbols, model_path='financial_report_model.h5'):
    bot = VNStockRecommender(model_path=model_path)
    X, y = [], []
    for s in symbols:
        df = bot.fetch_financial_report(s)
        X.append(df.values.flatten())
        pe = bot.fetch_pe(s)
        pb = bot.fetch_pb(s)
        y.append(1 if (pe < 15 and pb < 0.9) else 0)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    inp = Input(shape=(X.shape[1],))
    x = Dense(16, activation='relu')(inp)
    x = Dense(8, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2, callbacks=[es, mc])

if __name__ == '__main__':
    symbols = ['VIC', 'VHM', 'MSN', 'SHS', 'PVS', 'VND', 'GAS']
    train_financial_model(symbols)
    bot = VNStockRecommender()
    print("VN Stock Recommender Bot")
    while True:
        msg = input('Ban: ')
        if bot.normalize_text(msg) in ('exit', 'quit'):
            break
        print(bot.handle(msg))
