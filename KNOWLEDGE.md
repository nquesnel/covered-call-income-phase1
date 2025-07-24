# Covered Call Income System - Knowledge Base

## Project Overview
Building a comprehensive covered call income system to generate $2-5K monthly income to pay down $60K margin debt.

## System Architecture

### Core Components
1. **Position Manager** - Track stocks owned across accounts
2. **Opportunity Scanner** - Find best covered call opportunities
3. **Decision Tracker** - Record take/pass decisions with reasons
4. **21-50-7 Rule Alerts** - Alert when position drops 7% or time to roll
5. **Growth Scanner** - Find high-growth stocks before they explode
6. **WatchList** - Track interesting stocks before buying

### Data Storage
- JSON files in `/data` directory
- Positions, decisions, screening results, watchlist all stored as JSON
- Self-improving system tracks outcomes over time

## Key Trading Rules

### 21-50-7 Rule
- Sell calls 21+ days out
- Target 50%+ profit on calls
- Alert if stock drops 7% (consider exiting)

### Growth Categories
- **Conservative**: <15% YoY growth
- **Moderate**: 15-30% YoY growth  
- **Aggressive**: 30-60% YoY growth
- **Hypergrowth**: >60% YoY growth

### Position Sizing (Motley Fool Style)
- **Starter Position**: 1-2% of portfolio
- **Medium Position**: 3-5% of portfolio
- **Large Position**: 6-8% of portfolio
- **Conviction scores 80+**: Consider large positions

## Technical Implementation

### Technologies Used
- **Streamlit**: Web UI framework
- **yfinance**: Real-time market data
- **pandas**: Data manipulation
- **plotly**: Charting
- **tradingview-screener**: Dynamic market scanning

### API Integrations
- **Yahoo Finance** (via yfinance): Stock data, options chains
- **TradingView Screener**: Find all stocks by market cap
- **Unusual Whales API**: Considered but not yet implemented

### Known Issues & Solutions
1. **yfinance price fields**: Use multiple fallbacks (currentPrice, regularMarketPrice, price, history)
2. **TradingView syntax**: Use `.isin()` not `.in_()` for filters
3. **Virtual environment**: Must activate venv before running
4. **Parameter order**: save_json_data(filepath, data) not (data, filepath)

## Growth Screening Logic

### Three-Tier System
1. **Early Innings Scanner** ($1-50B market cap)
   - Find future 10-baggers in innings 1-3
   - High growth potential, emerging leaders
   - NOT for covered calls - buy and hold

2. **Tier 1: High Conviction** (5-15 stocks)
   - Strict criteria: >40% growth, PEG <1.5, etc.
   - Highest quality growth stocks
   - Limited/no covered calls

3. **Tier 2: Expanded Screen** (50-100 stocks)
   - Looser criteria for broader opportunities
   - Good for covered calls if growth <50%

### Screening Criteria
- Revenue growth (YoY and QoQ)
- Gross margins and trends
- Insider ownership
- Relative strength vs market
- Cash runway
- Industry tailwinds

### Self-Improvement System
- Tracks 30/90/180 day outcomes
- Logs successful/failed predictions
- Refines criteria based on results
- Stores history in screening_history.json

## Market Scanning

### Dynamic Universe
- No hard-coded ticker lists
- Scans entire market using TradingView
- Finds new IPOs automatically
- Caches results for 24 hours

### Fallback Methods
- Curated list of ~120 growth stocks
- S&P 500 components
- Manual symbol entry

## UI/UX Features

### Four Main Tabs
1. **Opportunities**: Scan positions for CC opportunities
2. **Positions**: Manage portfolio, add/remove stocks
3. **Growth Scanner**: Find new growth stocks
4. **WatchList**: Track stocks before buying

### Key Metrics Displayed
- Monthly income progress toward $3,500 goal
- Win rate on closed trades
- Active trade count
- Growth scores with explanations
- Conviction scores (0-100)
- Motley Fool style analysis

## Important File Paths
- Main app: `/Users/q/covered-call-income-phase1/app.py`
- Screening system: `/Users/q/covered-call-income-phase1/screening_system.py`
- Data directory: `/Users/q/covered-call-income-phase1/data/`
- Virtual environment: `/Users/q/covered-call-income-phase1/venv/`

## Running the System
```bash
cd /Users/q/covered-call-income-phase1
source venv/bin/activate
streamlit run app.py
```

Access at: http://localhost:8501

## Future Considerations
- Unusual Whales API integration for options flow
- Real-time alerts via Discord/SMS
- Backtesting capabilities
- Portfolio optimization
- Tax tracking integration

## Related Documentation
- **TODO.md** - Complete task list and roadmap
- **LEARNINGS.md** - Obstacles faced and insights gained
- **README.md** - Quick start guide

Last Updated: 2025-07-23 17:45 PST