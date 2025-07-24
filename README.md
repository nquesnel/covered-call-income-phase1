# Covered Call Income System

A comprehensive web application for managing covered call options strategies, tracking positions, and finding high-growth investment opportunities.

## Features

- **Position Manager**: Track your stock positions across multiple accounts
- **Opportunity Scanner**: Find optimal covered call opportunities based on the 21-50-7 rule
- **Growth Stock Scanner**: Discover high-growth stocks using multi-tier screening
- **WatchList**: Track potential investments before committing capital
- **Decision Tracking**: Record and learn from your trading decisions

## Live Demo

🚀 **[Launch App on Streamlit Cloud](https://covered-call-income.streamlit.app/)** (Coming soon)

## Quick Start (Local)

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/covered-call-income-phase1.git
cd covered-call-income-phase1
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open browser to http://localhost:8501

## Key Features

### 21-50-7 Rule
- Sell calls 21+ days to expiration
- Take profits at 50%
- Alert if underlying drops 7%

### Growth Screening
- **Early Innings Scanner**: Find future 10-baggers ($1-50B market cap)
- **Tier 1 High Conviction**: Strict criteria for best growth stocks
- **Tier 2 Expanded**: Broader screen for covered call candidates

### Dynamic Market Scanning
- No hard-coded ticker lists
- Automatically finds new IPOs and spin-offs
- Powered by TradingView screener

## Documentation

- [Knowledge Base](KNOWLEDGE.md) - System architecture and concepts
- [TODO List](TODO.md) - Roadmap and upcoming features
- [Learnings](LEARNINGS.md) - Obstacles faced and solutions

## Tech Stack

- **Frontend**: Streamlit
- **Data**: yfinance, TradingView Screener
- **Storage**: JSON files (local)
- **Charts**: Plotly
- **Deployment**: Streamlit Cloud

## Contributing

This is a personal project but suggestions are welcome! Please open an issue to discuss proposed changes.

## Disclaimer

This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.

## License

MIT License - see LICENSE file for details

---

Built with ❤️ for the retail investor community