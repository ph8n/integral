-- Market data: bars (OHLCV)
CREATE TABLE bars (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,           -- e.g. '1m', '1h', '1d'
    open_time TIMESTAMPTZ NOT NULL,          -- start time of bar
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL
);
CREATE INDEX idx_bars_symbol_time ON bars(symbol, open_time);

-- Orders
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    order_type VARCHAR(10) NOT NULL,
    quantity NUMERIC NOT NULL,
    price NUMERIC,
    status VARCHAR(20) NOT NULL,
    filled_quantity NUMERIC,
    avg_fill_price NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fills (trades)
CREATE TABLE fills (
    id BIGSERIAL PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    symbol VARCHAR(20) NOT NULL,
    quantity NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    side VARCHAR(4) NOT NULL,
    commission NUMERIC,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Positions
CREATE TABLE positions (
    symbol VARCHAR(20) PRIMARY KEY,
    quantity NUMERIC NOT NULL,
    avg_cost NUMERIC NOT NULL,
    realized_pnl NUMERIC DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
