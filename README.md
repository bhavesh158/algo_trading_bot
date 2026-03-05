# Automated AI Trading System

Two independent automated trading systems in one repo:

- **Stocks** — Intraday trading on NSE (Nifty 50), session-based (9:15 AM – 3:30 PM IST)
- **Crypto** — 24/7 cryptocurrency trading via Binance (BTC, ETH, SOL, DOGE, and more)

Both support **paper trading** (simulated) and **live trading** via broker/exchange APIs.

---

## Quick Start (Local Development)

```bash
# Clone and enter the project
cd algo_trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install stock dependencies
pip install -r requirements.txt

# Install crypto dependencies
pip install -r crypto/requirements.txt

# Run stock trading (paper mode)
python -m stocks.main --mode paper

# Run crypto trading (paper mode)
python -m crypto.crypto_main --mode paper
```

---

## Running in Production

### Option 1: Docker (Recommended)

Docker packages everything — Python, dependencies, code — into a single image. Only Docker and Docker Compose are required on the server.

**Setup:**

```bash
# 1. Get the code on your server
git clone <your-repo-url> ~/algo_trading_bot
cd ~/algo_trading_bot

# 2. Create .env with your config
cp .env.example .env
nano .env   # Edit with your values

# 3. Build and start
docker compose build
docker compose up -d
```

**Managing the container:**

```bash
# View live logs
docker compose logs -f

# Check status
docker compose ps

# Graceful stop (closes all open positions, waits up to 30s)
docker compose stop

# Restart
docker compose restart

# View state/journal from the volume
docker compose exec crypto-trader cat /app/crypto_data/state.json
docker compose exec crypto-trader cat /app/crypto_data/trade_journal.csv

# Rebuild after code changes
docker compose build && docker compose up -d
```

**What happens on stop/crash:**

| Scenario | Docker Behavior | Bot Behavior |
|---|---|---|
| `docker compose stop` | Sends SIGTERM, waits 30s | Closes all positions at market → clears state → exits |
| `docker compose restart` | Stop then start | Graceful close, then fresh start |
| Server reboot | Auto-restarts (`unless-stopped`) | Restores positions from `state.json` |
| Container crash / OOM | Auto-restarts | Restores positions from `state.json` |
| `docker compose down` | Stops + removes container (volume kept) | Graceful shutdown |
| `docker compose down -v` | Removes container + volume | **All data lost** — don't do this with open positions |

### Option 2: systemd Service (Without Docker)

If you prefer running directly on the host without Docker:

**1. Set up the project:**

```bash
cd ~/algo_trading_bot
python3 -m venv venv
source venv/bin/activate
pip install -r crypto/requirements.txt
cp .env.example .env
nano .env
```

**2. Create the systemd service file:**

```bash
sudo nano /etc/systemd/system/crypto-trader.service
```

Paste:

```ini
[Unit]
Description=Crypto Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bhavesh
WorkingDirectory=/home/bhavesh/algo_trading_bot
EnvironmentFile=/home/bhavesh/algo_trading_bot/.env
ExecStart=/home/bhavesh/algo_trading_bot/venv/bin/python -m crypto.crypto_main --mode paper
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGTERM

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trader

[Install]
WantedBy=multi-user.target
```

**3. Enable and start:**

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start the service
sudo systemctl start crypto-trader

# Enable auto-start on boot
sudo systemctl enable crypto-trader

# Check status
sudo systemctl status crypto-trader

# View logs
journalctl -u crypto-trader -f

# Graceful stop (closes positions)
sudo systemctl stop crypto-trader

# Restart
sudo systemctl restart crypto-trader
```

---

## Status API (Monitoring)

A lightweight HTTP API for monitoring the bot remotely. **Disabled by default** — no port is opened unless you explicitly enable it.

**Enable in `.env`:**

```bash
CRYPTO_TRADING__api__enabled=true
CRYPTO_TRADING__api__port=8599    # Change if port conflicts
```

If using Docker, also set the port mapping:

```bash
CRYPTO_API_PORT=8599   # Must match the port above
```

**Endpoints:**

```bash
# Liveness check
curl http://your-server:8599/health

# Open positions, capital, P&L
curl http://your-server:8599/status

# Recent trade history
curl http://your-server:8599/trades
```

**Example `/status` response:**

```json
{
  "capital": {
    "total": 1000.0,
    "available": 662.85,
    "equity": 998.03,
    "peak": 1000.0,
    "drawdown_pct": 0.2
  },
  "pnl": {
    "realized": -1.97,
    "unrealized": 0.0,
    "total": -1.97
  },
  "positions": {
    "count": 4,
    "details": [...]
  },
  "active_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
}
```

Uses Python's stdlib `http.server` — zero extra dependencies.

---

## Configuration

### Environment Variables (`.env`)

Credentials and config overrides are loaded from `.env` in the project root. This file is git-ignored.

```bash
cp .env.example .env
```

**Crypto exchange credentials:**

| Variable | Description |
|---|---|
| `EXCHANGE_API_KEY` | Binance API key |
| `EXCHANGE_API_SECRET` | Binance API secret |
| `EXCHANGE_PASSWORD` | Exchange password (if required) |

**Stock broker credentials (Angel One):**

| Variable | Description |
|---|---|
| `BROKER_API_KEY` | API key from [SmartAPI portal](https://smartapi.angelone.in/) |
| `BROKER_CLIENT_ID` | Angel One client ID (e.g. `A12345`) |
| `BROKER_PASSWORD` | Trading password |
| `BROKER_TOTP_SECRET` | TOTP secret for 2FA |

**Config overrides via env vars:**

Any YAML config value can be overridden using environment variables:

```bash
# Crypto overrides (CRYPTO_TRADING__section__key)
CRYPTO_TRADING__account__initial_capital=5000
CRYPTO_TRADING__risk__max_risk_per_trade_pct=1.0
CRYPTO_TRADING__selection__min_24h_volume_usdt=1000000
CRYPTO_TRADING__strategies__default_confidence_threshold=0.4

# Stock overrides (ALGO_TRADING__section__key)
ALGO_TRADING__account__initial_capital=200000
ALGO_TRADING__risk__max_risk_per_trade_pct=0.5
```

### YAML Config

Default settings:
- **Stocks:** `stocks/config/default_config.yaml`
- **Crypto:** `crypto/config/default_config.yaml`

Pass a custom config file to override specific values:

```bash
python -m crypto.crypto_main --mode paper --config my_config.yaml
```

---

## Crypto Trading System

### Strategies

Three strategies run in parallel on each active pair:

- **Trend Following** — EMA crossovers (9/21) with ADX confirmation. Generates both long and short signals. Fires on trend continuation (ADX > 40) without requiring exact crossover candle.
- **Mean Reversion** — Bollinger Band + RSI. Buys at lower band / oversold RSI, sells at upper band / overbought RSI.
- **Breakout Momentum** — Volume-confirmed breakouts above/below recent highs/lows.

### Pair Selection

The system dynamically selects the best trading pairs from 20 candidates based on:
- 24h volume (default: > $10M USDT)
- Bid-ask spread (default: < 0.15%)
- Order book depth (default: > $50K each side)

Pairs with open positions are always protected from being filtered out, regardless of filter criteria.

Refreshed every 4 hours.

### State Persistence & Crash Recovery

The system saves its state to `crypto_data/state.json` on every position open/close:
- **Graceful shutdown** (Ctrl+C / SIGTERM / SIGHUP): Closes all open positions at market price, logs to trade journal, clears state, exits.
- **Crash / kill -9 / power loss**: Positions are preserved in `state.json`. On restart, they are automatically restored and the system continues tracking them.
- **Partial shutdown failure**: If some positions can't be closed (e.g., exchange unreachable), those positions remain in state for recovery on next start.

### Trade Journal

All position opens and closes are logged to `crypto_data/trade_journal.csv` — a permanent, append-only record for post-analysis.

### Risk Management

- Max 1% capital risked per trade
- Max 5 simultaneous positions
- Rolling 24h loss limit: 5% (pauses trading)
- Max 50% total capital exposure
- Max 20% single pair exposure
- Drawdown protection: reduces size at 5%, pauses at 8%
- Volatility safeguards: pauses if any pair moves > 5% in 15 minutes

---

## Stock Trading System (NSE)

### Daily Lifecycle (IST)

1. **Pre-Market (8:45 – 9:15)** — Loads daily data, scores Nifty 50 stocks, builds watchlist
2. **Market Hours (9:15 – 15:15)** — Runs strategies, generates signals, executes orders
3. **Pre-Close (15:15 – 15:30)** — Squares off all intraday positions
4. **Post-Market** — Generates daily report

### Strategies

- **Mean Reversion** — Trades price deviations from moving average (z-score based)
- **Momentum Breakout** — Volume-confirmed breakouts with ATR-based stops/targets
- **Opening Range Breakout (ORB)** — Trades breakouts of first 15 minutes' high/low

### Setting Up Angel One

1. Create an account at [Angel One](https://www.angelone.in/)
2. Register for SmartAPI at [smartapi.angelone.in](https://smartapi.angelone.in/)
3. Get your API key and enable TOTP
4. Add credentials to `.env`
5. Run: `python -m stocks.main --mode live`

---

## Project Structure

```
algo_trading_bot/
├── stocks/                          # NSE stock trading system
│   ├── main.py                      # Stock CLI entrypoint
│   ├── config/                      # YAML config & settings
│   ├── core/                        # Enums, models, event bus
│   ├── data/                        # Market data (Yahoo Finance)
│   ├── selection/                   # Nifty 50 stock selector
│   ├── strategy/                    # Mean reversion, momentum, ORB
│   ├── analysis/                    # AI analysis, regime detection
│   ├── risk/                        # Risk manager, position sizer
│   ├── execution/                   # Paper trader, Angel One, Zerodha
│   ├── portfolio/                   # Portfolio & performance tracking
│   ├── scheduler/                   # NSE session scheduler
│   ├── reporting/                   # Reports & alerts
│   └── utils/                       # Logging, security
├── crypto/                          # 24/7 crypto trading system
│   ├── crypto_main.py               # Crypto CLI entrypoint
│   ├── config/                      # Crypto config & default YAML
│   ├── core/                        # Models, enums, event bus
│   ├── data/                        # Exchange data via ccxt
│   ├── selection/                   # Pair selector & scoring
│   ├── strategy/                    # Trend following, mean reversion, breakout
│   ├── analysis/                    # Regime detection, AI, volatility
│   ├── risk/                        # Risk manager, position sizer, drawdown
│   ├── execution/                   # Paper & live exchange execution
│   ├── portfolio/                   # Portfolio manager, state persistence
│   ├── scheduler/                   # 24/7 continuous scheduler
│   ├── reporting/                   # Reports, alerts, trade journal
│   ├── api/                         # Status API server (optional)
│   └── utils/                       # Logging, security
├── Dockerfile                       # Docker image for crypto system
├── docker-compose.yml               # Docker Compose config
├── .dockerignore
├── .env.example                     # Environment variable template
├── .gitignore
├── requirements.txt                 # Stock system dependencies
└── README.md
```

---

## CLI Options

```bash
# Stock trading
python -m stocks.main --mode paper              # Paper trading (default)
python -m stocks.main --mode live               # Live trading
python -m stocks.main --config my_config.yaml   # Custom config
python -m stocks.main --log-level DEBUG         # Verbose logging

# Crypto trading
python -m crypto.crypto_main --mode paper       # Paper trading (default)
python -m crypto.crypto_main --mode live        # Live trading
python -m crypto.crypto_main --config conf.yaml # Custom config
python -m crypto.crypto_main --log-level DEBUG  # Verbose logging
```

---

## Troubleshooting

**Crypto: No trades happening**
The system only trades when strategy signals meet the confidence threshold AND pass all risk checks. Lower the threshold with `CRYPTO_TRADING__strategies__default_confidence_threshold=0.4` for testing.

**Crypto: Too few pairs selected**
Lower the volume filter: `CRYPTO_TRADING__selection__min_24h_volume_usdt=1000000`

**Crypto: Positions lost after restart**
Check if `crypto_data/state.json` exists. If the process was killed with `kill -9`, positions should be in the state file and restored on restart. If the file is missing, the previous shutdown cleared it (meaning positions were closed normally).

**Stocks: No trades during market hours**
The system only trades during NSE hours (9:15 AM – 3:30 PM IST, Mon–Fri). Check that your watchlist has symbols and strategies are generating signals above the confidence threshold.

**Angel One login fails**
Verify your TOTP secret is the base32 secret from SmartAPI setup, not the 6-digit code.

**Docker: Can't access status API**
Make sure both `CRYPTO_TRADING__api__enabled=true` and port mapping are set in `.env`. Check with `docker compose logs` for "Status API server started" message.

**Docker: Data missing after rebuild**
Data lives in the `crypto_data` Docker volume, which persists across `docker compose down`. Only `docker compose down -v` removes it.
