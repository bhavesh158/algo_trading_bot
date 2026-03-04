# Automated AI Trading System

An automated intraday trading system for the **Indian stock market (NSE)**. It selects stocks from the Nifty 50 universe, runs multiple trading strategies, manages risk, and executes orders — all autonomously during market hours.

Supports **paper trading** (simulated) and **live trading** via Angel One or Zerodha.

---

## Quick Start

```bash
# 1. Clone and enter the project
cd algo_trading_bot

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run in paper trading mode (no broker account needed)
python main.py --mode paper
```

That's it — the system will start paper trading during NSE market hours (9:15 AM – 3:30 PM IST).

---

## Configuration

### Environment Variables (`.env`)

Credentials and sensitive config are loaded from a **`.env`** file in the project root.

```bash
# Create your .env from the template
cp .env.example .env
```

Then edit `.env` with your values. The file is git-ignored and never committed.

#### Angel One credentials

| Variable | Description |
|---|---|
| `BROKER_API_KEY` | API key from [SmartAPI portal](https://smartapi.angelone.in/) |
| `BROKER_CLIENT_ID` | Your Angel One client ID (e.g. `A12345`) |
| `BROKER_PASSWORD` | Your trading password |
| `BROKER_TOTP_SECRET` | TOTP secret for 2FA (from authenticator app setup) |

#### Zerodha credentials (if using Zerodha)

| Variable | Description |
|---|---|
| `BROKER_API_KEY` | Kite Connect API key |
| `BROKER_API_SECRET` | Kite Connect API secret |
| `BROKER_ACCESS_TOKEN` | Session access token |

#### Config overrides via env vars

Any config value can be overridden using env vars with the `ALGO_TRADING__` prefix:

```bash
# In .env — override capital to ₹2,00,000
ALGO_TRADING__account__initial_capital=200000

# Reduce max risk per trade to 0.5%
ALGO_TRADING__risk__max_risk_per_trade_pct=0.5
```

### YAML Config

All default settings live in `config/default_config.yaml`. You can also pass a custom config file:

```bash
python main.py --mode paper --config my_config.yaml
```

The custom file is merged on top of defaults — you only need to include the values you want to change.

**Key settings:**

- `account.initial_capital` — Starting capital in INR (default: `100000`)
- `broker.adapter` — Broker to use: `angelone` or `zerodha`
- `risk.max_risk_per_trade_pct` — Max capital risked per trade (default: `1.0%`)
- `risk.max_open_positions` — Max simultaneous positions (default: `5`)
- `risk.max_daily_loss_pct` — Stop trading if daily loss exceeds this (default: `3.0%`)
- `selection.max_watchlist_size` — Number of stocks to track (default: `20`)

---

## Trading Modes

### Paper Trading (default)

Simulates order execution with configurable slippage and commissions. No broker account required.

```bash
python main.py --mode paper
```

**Recommended:** Run in paper mode for at least 2–4 weeks to validate strategy performance before going live.

### Live Trading

Executes real orders through your broker account.

```bash
# 1. Fill in your .env with broker credentials
# 2. Set broker adapter in config/default_config.yaml:
#    broker.adapter: angelone   (or zerodha)
# 3. Run
python main.py --mode live
```

**Pre-flight checks:** The system validates that all required credentials are present before starting in live mode. Missing credentials will cause a clear error at startup.

---

## Setting Up Angel One

1. **Create an account** at [Angel One](https://www.angelone.in/) (includes Demat + trading account)
2. **Register for SmartAPI** at [smartapi.angelone.in](https://smartapi.angelone.in/)
3. **Get your API key** from the SmartAPI dashboard
4. **Set up TOTP:** In the SmartAPI portal, enable TOTP and copy the secret key
5. **Add credentials to `.env`:**
   ```
   BROKER_API_KEY=your_api_key_here
   BROKER_CLIENT_ID=A12345
   BROKER_PASSWORD=your_trading_password
   BROKER_TOTP_SECRET=your_totp_secret_here
   ```
6. **Set broker in config:** Ensure `broker.adapter: angelone` in `config/default_config.yaml` (this is the default)
7. **Run:** `python main.py --mode live`

---

## How It Works

### Daily Lifecycle

The system follows a strict session schedule (all times IST):

1. **Pre-Market (8:45 – 9:15)** — Loads daily data, scores Nifty 50 stocks, builds a watchlist of the top 20 by volume/liquidity
2. **Market Hours (9:15 – 15:15)** — Fetches live candle data, runs strategies, generates signals, checks risk rules, and executes orders
3. **Pre-Close (15:15 – 15:30)** — Squares off all open intraday positions
4. **Post-Market** — Generates daily performance report (saved to `reports/`)

If the system starts mid-session (e.g., at 10:00 AM), it auto-runs the pre-market setup before entering market hours.

### Strategies

Three strategies run in parallel, each generating independent signals:

- **Mean Reversion** — Trades when price deviates significantly from its moving average (z-score based entry/exit)
- **Momentum Breakout** — Enters on volume-confirmed breakouts above recent highs, with ATR-based stops and targets
- **Opening Range Breakout (ORB)** — Trades breakouts of the first 15 minutes' high/low range

### Signal Flow

```
Market Data → Strategies → AI Confidence Adjustment → Risk Check → Position Sizing → Order Execution
```

Each signal goes through:
1. **Multi-timeframe analysis** — Confirms the signal across 1m, 5m, 15m candles
2. **Market regime detection** — Adjusts behavior for trending vs. ranging markets
3. **AI confidence scoring** — Adjusts signal confidence based on technical patterns
4. **Risk manager** — Checks daily loss limits, exposure limits, and position count
5. **Position sizer** — Calculates quantity using volatility-adjusted sizing
6. **Order executor** — Routes to paper trader or live broker

### Risk Management

- Max 1% of capital risked per trade
- Max 5 simultaneous positions
- Stops trading if daily loss exceeds 3%
- Max 50% of total capital deployed at any time
- Automatic drawdown protection (reduces size at 3%, pauses at 5%)
- Volatility safeguards — pauses if Nifty moves > 3% intraday

---

## Project Structure

```
algo_trading_bot/
├── main.py                          # CLI entrypoint & orchestrator
├── config/
│   ├── settings.py                  # YAML + .env config loader
│   └── default_config.yaml          # All default settings
├── core/
│   ├── enums.py                     # OrderSide, OrderType, TradingMode, etc.
│   ├── models.py                    # Candle, Signal, Order, Position, Trade
│   ├── events.py                    # Event types (SignalEvent, OrderEvent, etc.)
│   └── event_bus.py                 # Publish-subscribe event bus
├── data/
│   ├── market_data_engine.py        # Candle fetching + technical indicators
│   ├── data_provider.py             # Abstract data provider interface
│   └── providers/
│       └── yahoo_provider.py        # Yahoo Finance data provider
├── selection/
│   └── stock_selector.py            # Nifty 50 scoring & watchlist builder
├── strategy/
│   ├── base_strategy.py             # Abstract strategy interface
│   ├── strategy_engine.py           # Runs all strategies, aggregates signals
│   └── strategies/
│       ├── mean_reversion.py
│       ├── momentum_breakout.py
│       └── opening_range_breakout.py
├── analysis/
│   ├── ai_analysis.py               # AI-based confidence adjustment
│   ├── regime_detector.py           # Trending / ranging / volatile detection
│   ├── multi_timeframe.py           # Cross-timeframe signal confirmation
│   └── news_monitor.py              # Volatility safeguards
├── risk/
│   ├── risk_manager.py              # Trade approval, exposure checks
│   ├── position_sizer.py            # Volatility-adjusted position sizing
│   └── drawdown_monitor.py          # Drawdown alerts & auto-protection
├── execution/
│   ├── order_executor.py            # Routes orders (paper / live)
│   ├── paper_trader.py              # Simulated execution with slippage
│   └── broker_adapters/
│       ├── base_adapter.py          # Abstract broker interface
│       ├── angelone_adapter.py      # Angel One SmartAPI adapter
│       └── zerodha_adapter.py       # Zerodha Kite adapter
├── portfolio/
│   ├── portfolio_manager.py         # Tracks positions, P&L, capital
│   └── performance_monitor.py       # Win rate, Sharpe, drawdown metrics
├── scheduler/
│   └── trading_scheduler.py         # Session controller (pre-market → close)
├── reporting/
│   ├── report_generator.py          # Daily JSON reports
│   └── alert_manager.py             # Console/file alerts on trades & errors
├── utils/
│   ├── logger.py                    # Logging setup
│   └── security.py                  # Credential validation
├── .env.example                     # Environment variable template
├── .gitignore
└── requirements.txt
```

---

## CLI Options

```bash
python main.py --mode paper              # Paper trading (default)
python main.py --mode live               # Live trading
python main.py --config my_config.yaml   # Custom config file
python main.py --log-level DEBUG         # Override log level
```

---

## Reports

Daily performance reports are saved to the `reports/` directory in JSON format after each trading session. Reports include:

- Trades executed (entry/exit prices, P&L)
- Portfolio value over time
- Win rate, average profit/loss
- Max drawdown
- Strategy-wise performance breakdown

---

## Troubleshooting

**System starts but no trades happen:**
The system only trades during NSE market hours (9:15 AM – 3:30 PM IST, Mon–Fri). If no strategy signals meet the confidence threshold and pass risk checks, no trades are placed — this is expected and safe behavior.

**"0 symbols in watchlist":**
This happens if daily data couldn't be fetched for the Nifty 50 stocks. Check your internet connection. The system retries on the next cycle.

**Angel One login fails:**
Verify your TOTP secret is correct. It should be the base32 secret from the SmartAPI setup, not the 6-digit code.

**Live mode refuses to start:**
The system checks all required env vars before starting. Check the error messages — they'll tell you exactly which variable is missing.
