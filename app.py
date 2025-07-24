import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from screening_system import GrowthScreeningSystem

st.set_page_config(
    page_title="Covered Call Income System",
    page_icon="📈",
    layout="wide"
)

DATA_DIR = "data"
POSITIONS_FILE = os.path.join(DATA_DIR, "positions.json")
DECISIONS_FILE = os.path.join(DATA_DIR, "trade_decisions.json")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_json_data(filepath: str) -> Dict:
    """Load data from JSON file, return empty dict if file doesn't exist"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_json_data(filepath: str, data: Dict) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def initialize_session_state():
    """Initialize session state variables"""
    if 'positions' not in st.session_state:
        st.session_state.positions = load_json_data(POSITIONS_FILE)
    if 'decisions' not in st.session_state:
        st.session_state.decisions = load_json_data(DECISIONS_FILE)
    if 'growth_screener' not in st.session_state:
        st.session_state.growth_screener = GrowthScreeningSystem()

def check_21_50_7_alerts() -> List[Dict]:
    """Check for 21-50-7 rule violations"""
    alerts = []
    decisions = st.session_state.decisions
    
    for trade_id, trade in decisions.items():
        if trade.get('status') != 'ACTIVE':
            continue
            
        exp_date = datetime.strptime(trade['expiration'], '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        
        if trade.get('current_profit_pct', 0) >= 50:
            alerts.append({
                'type': 'CLOSE_NOW',
                'symbol': trade['symbol'],
                'strike': trade['strike'],
                'expiration': trade['expiration'],
                'profit_pct': trade.get('current_profit_pct', 0),
                'message': f"🚨 {trade['symbol']} ${trade['strike']} - At 50% profit! CLOSE NOW"
            })
        elif days_to_exp <= 7:
            alerts.append({
                'type': 'MUST_CLOSE',
                'symbol': trade['symbol'],
                'strike': trade['strike'],
                'expiration': trade['expiration'],
                'days_left': days_to_exp,
                'message': f"⚠️ {trade['symbol']} ${trade['strike']} - {days_to_exp} days left! HIGH GAMMA RISK"
            })
        elif days_to_exp <= 21:
            alerts.append({
                'type': 'MONITOR',
                'symbol': trade['symbol'],
                'strike': trade['strike'],
                'expiration': trade['expiration'],
                'days_left': days_to_exp,
                'message': f"👀 {trade['symbol']} ${trade['strike']} - {days_to_exp} days to expiration"
            })
    
    return alerts

def display_alerts(alerts: List[Dict]):
    """Display critical alerts at top of page"""
    if alerts:
        st.markdown("### 🚨 CRITICAL ALERTS 🚨")
        for alert in alerts:
            if alert['type'] == 'CLOSE_NOW':
                st.error(alert['message'])
            elif alert['type'] == 'MUST_CLOSE':
                st.warning(alert['message'])
            else:
                st.info(alert['message'])

def display_monthly_progress():
    """Display monthly income progress"""
    decisions = st.session_state.decisions
    current_month = datetime.now().strftime('%Y-%m')
    
    monthly_income = 0
    active_trades = 0
    win_count = 0
    total_trades = 0
    
    for trade in decisions.values():
        trade_month = trade['timestamp'][:7]
        if trade_month == current_month:
            if trade['status'] == 'CLOSED':
                monthly_income += trade.get('profit_loss', 0)
                total_trades += 1
                if trade.get('profit_loss', 0) > 0:
                    win_count += 1
            elif trade['status'] == 'ACTIVE':
                active_trades += 1
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Income Goal", "$3,500", f"${monthly_income:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Active Trades", active_trades)
    with col4:
        st.metric("Margin Debt", "$60,000", "-$2,800")

def add_position():
    """Add new position form"""
    st.subheader("Add New Position")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", placeholder="TSLA")
        shares = st.number_input("Shares", min_value=0, value=100, step=100)
        cost_basis = st.number_input("Cost Basis", min_value=0.0, value=100.0, step=0.01)
    
    with col2:
        account_type = st.selectbox("Account Type", ["taxable", "roth"])
        
        # Growth Score Calculator
        if symbol:
            if st.button("📊 Calculate Growth Score", type="secondary"):
                with st.spinner(f"Analyzing {symbol.upper()}..."):
                    score, category, analysis = calculate_growth_score(symbol.upper())
                    st.session_state.growth_analysis = analysis
                    st.session_state.suggested_category = category
        
        # Show analysis if available
        if hasattr(st.session_state, 'growth_analysis'):
            analysis = st.session_state.growth_analysis
            
            # Display score with color
            score_color = "🔴" if analysis['score'] >= 75 else "🟡" if analysis['score'] >= 50 else "🟢"
            st.metric("Growth Score", f"{score_color} {analysis['score']}/100")
            
            # Show recommendation
            st.info(analysis['recommendation'])
            
            # Show factors
            with st.expander("📋 Score Breakdown"):
                for factor, value in analysis['factors'].items():
                    st.write(f"**{factor}:** {value}")
        
        # Growth category selector with suggested default
        default_category = getattr(st.session_state, 'suggested_category', 'MODERATE')
        growth_category = st.selectbox(
            "Growth Category",
            ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
            index=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"].index(default_category),
            help="Based on growth score analysis above"
        )
        
        notes = st.text_area("Notes", placeholder="High growth - protect upside")
    
    if st.button("Add Position", type="primary"):
        if symbol and shares > 0:
            symbol = symbol.upper()  # Convert to uppercase here
            position_key = f"{symbol}_{account_type.upper()}"
            
            st.session_state.positions[position_key] = {
                "symbol": symbol,
                "shares": shares,
                "cost_basis": cost_basis,
                "account_type": account_type,
                "position_key": position_key,
                "notes": notes,
                "growth_category": growth_category
            }
            
            save_json_data(POSITIONS_FILE, st.session_state.positions)
            
            # Clear the growth analysis from session state
            if hasattr(st.session_state, 'growth_analysis'):
                del st.session_state.growth_analysis
            if hasattr(st.session_state, 'suggested_category'):
                del st.session_state.suggested_category
                
            st.success(f"Added {symbol} position!")
            st.rerun()

def display_positions():
    """Display current positions"""
    st.subheader("Current Positions")
    
    if not st.session_state.positions:
        st.info("No positions yet. Add your first position above!")
        return
    
    positions_df = pd.DataFrame.from_dict(
        st.session_state.positions, 
        orient='index'
    )
    
    for idx, row in positions_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.write(f"**{row['symbol']}** ({row['account_type'].upper()})")
        with col2:
            st.write(f"{row['shares']} shares")
        with col3:
            st.write(f"${row['cost_basis']:.2f}")
        with col4:
            st.write(row['growth_category'])
        with col5:
            if st.button(f"Delete", key=f"del_{idx}"):
                del st.session_state.positions[idx]
                save_json_data(POSITIONS_FILE, st.session_state.positions)
                st.rerun()

def calculate_iv_rank(symbol: str, current_iv: float) -> float:
    """Calculate IV rank (simplified for Phase 1)"""
    return min(current_iv * 100 / 30, 100)

def calculate_growth_score(symbol: str) -> tuple[int, str, dict]:
    """Calculate growth score based on real market data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Initialize scores
        score = 0  # Start at 0, not 50
        factors = {}
        
        # 1. Revenue Growth (0-25 points)
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.50:  # >50% growth
            revenue_points = 25
        elif revenue_growth > 0.30:  # >30% growth
            revenue_points = 20
        elif revenue_growth > 0.20:  # >20% growth
            revenue_points = 15
        elif revenue_growth > 0.10:  # >10% growth
            revenue_points = 10
        else:
            revenue_points = max(0, revenue_growth * 50)  # Scale smaller growth
        
        score += revenue_points
        if revenue_growth:
            factors['Revenue Growth'] = f"{revenue_growth:.1%}"
        
        # 2. Forward PE Ratio (0-20 points) - High PE = growth stock
        pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
        if pe_ratio > 50:
            pe_points = 20
            factors['Valuation'] = f"PE {pe_ratio:.1f} - Extreme growth premium"
        elif pe_ratio > 35:
            pe_points = 15
            factors['Valuation'] = f"PE {pe_ratio:.1f} - High growth premium"
        elif pe_ratio > 25:
            pe_points = 10
            factors['Valuation'] = f"PE {pe_ratio:.1f} - Growth stock"
        elif pe_ratio > 15:
            pe_points = 5
            factors['Valuation'] = f"PE {pe_ratio:.1f} - Fair value"
        else:
            pe_points = 0
            factors['Valuation'] = f"PE {pe_ratio:.1f} - Value stock"
        score += pe_points
        
        # 3. Price Performance (0-20 points) - Momentum indicator
        hist = ticker.history(period="1y")
        if not hist.empty:
            year_ago_price = hist['Close'].iloc[0]
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            year_return = ((current_price - year_ago_price) / year_ago_price) * 100
            
            if year_return > 100:  # >100% gain
                perf_points = 20
            elif year_return > 50:  # >50% gain
                perf_points = 15
            elif year_return > 25:  # >25% gain
                perf_points = 10
            elif year_return > 10:  # >10% gain
                perf_points = 5
            else:
                perf_points = 0
            
            score += perf_points
            factors['1Y Performance'] = f"{year_return:.1f}%"
        
        # 4. Analyst Sentiment (0-20 points)
        target_price = info.get('targetMeanPrice', 0)
        current_price = info.get('currentPrice', 0)
        if target_price and current_price:
            upside = ((target_price - current_price) / current_price) * 100
            if upside > 50:
                analyst_points = 20
            elif upside > 30:
                analyst_points = 15
            elif upside > 20:
                analyst_points = 10
            elif upside > 10:
                analyst_points = 5
            else:
                analyst_points = 0
            
            score += analyst_points
            factors['Analyst Target'] = f"{upside:+.1f}% upside"
        
        # 5. Sector and Market Cap Bonus (0-15 points)
        sector = info.get('sector', '')
        market_cap = info.get('marketCap', 0)
        
        sector_points = 0
        if sector in ['Technology', 'Consumer Cyclical', 'Communication Services']:
            sector_points += 10
            factors['Sector'] = f"{sector} (growth sector)"
        else:
            factors['Sector'] = sector
            
        # High-growth companies often have large market caps
        if market_cap > 1_000_000_000_000:  # >$1T
            sector_points += 5
            factors['Market Cap'] = "Mega cap ($1T+)"
        elif market_cap > 100_000_000_000:  # >$100B
            sector_points += 3
            factors['Market Cap'] = "Large cap ($100B+)"
        
        score += sector_points
        
        # Special cases for known high-growth stocks
        high_growth_stocks = ['TSLA', 'NVDA', 'PLTR', 'NET', 'DDOG', 'SNOW', 'CRWD', 'PANW']
        if symbol.upper() in high_growth_stocks:
            score = max(score, 75)  # Minimum score of 75 for known growth stocks
            factors['Special'] = "Known high-growth stock"
        
        # Cap score at 100
        score = min(100, max(0, score))
        
        # Determine category with better thresholds
        if score >= 70:
            category = "AGGRESSIVE"
            recommendation = "🚨 HIGH GROWTH - Protect upside! Only sell far OTM calls or skip entirely."
        elif score >= 40:
            category = "MODERATE"
            recommendation = "⚖️ BALANCED - Standard 5-10% OTM covered calls work well."
        else:
            category = "CONSERVATIVE"
            recommendation = "✅ VALUE/DIVIDEND PLAY - Can sell aggressive ATM or slightly OTM calls."
        
        return score, category, {
            'score': score,
            'category': category,
            'recommendation': recommendation,
            'factors': factors
        }
        
    except Exception as e:
        # If API fails, return moderate default
        return 50, "MODERATE", {
            'score': 50,
            'category': 'MODERATE',
            'recommendation': '⚖️ Unable to calculate - defaulting to moderate strategy',
            'factors': {'Error': str(e)}
        }

def get_growth_score(symbol: str, growth_category: str) -> int:
    """Get growth score based on category (simplified for Phase 1)"""
    scores = {
        "CONSERVATIVE": 25,
        "MODERATE": 50,
        "AGGRESSIVE": 75
    }
    return scores.get(growth_category, 50)

def scan_covered_call_opportunities():
    """Scan positions for covered call opportunities"""
    opportunities = []
    
    for position_key, position in st.session_state.positions.items():
        try:
            ticker = yf.Ticker(position['symbol'])
            current_price = ticker.info.get('currentPrice', 0)
            
            if current_price == 0:
                continue
            
            options_dates = ticker.options
            if not options_dates:
                continue
            
            target_date = datetime.now() + timedelta(days=30)
            best_date = min(options_dates, 
                          key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
            
            opt_chain = ticker.option_chain(best_date)
            calls = opt_chain.calls
            
            otm_calls = calls[calls['strike'] > current_price * 1.05]
            
            if not otm_calls.empty:
                for _, call in otm_calls.head(3).iterrows():
                    iv = call.get('impliedVolatility', 0.3)
                    iv_rank = calculate_iv_rank(position['symbol'], iv)
                    
                    if iv_rank > 50:
                        growth_score = get_growth_score(
                            position['symbol'], 
                            position['growth_category']
                        )
                        
                        premium = (call['bid'] + call['ask']) / 2
                        monthly_yield = (premium / current_price) * 100
                        
                        opportunities.append({
                            'position_key': position_key,
                            'symbol': position['symbol'],
                            'current_price': current_price,
                            'strike': call['strike'],
                            'expiration': best_date,
                            'premium': premium,
                            'bid': call['bid'],
                            'ask': call['ask'],
                            'iv': iv,
                            'iv_rank': iv_rank,
                            'growth_score': growth_score,
                            'monthly_yield': monthly_yield,
                            'shares': position['shares'],
                            'max_contracts': position['shares'] // 100,
                            'growth_category': position['growth_category']
                        })
                        
        except Exception as e:
            st.error(f"Error scanning {position['symbol']}: {str(e)}")
    
    return sorted(opportunities, key=lambda x: x['monthly_yield'], reverse=True)

def display_opportunity_card(opp: Dict):
    """Display opportunity card with take/pass buttons"""
    days_to_exp = (datetime.strptime(opp['expiration'], '%Y-%m-%d') - datetime.now()).days
    
    with st.container():
        st.markdown(f"""
        ### {opp['symbol']} - {opp['growth_category']} STRATEGY
        **Strike:** ${opp['strike']:.2f} | **Premium:** ${opp['premium']:.2f} | **Exp:** {opp['expiration']} ({days_to_exp}d)  
        **Growth Score:** {opp['growth_score']}/100 | **IV Rank:** {opp['iv_rank']:.0f}% | **Monthly Yield:** {opp['monthly_yield']:.1f}%
        """)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("✅ TAKE", key=f"take_{opp['position_key']}_{opp['strike']}"):
                st.session_state.show_take_dialog = opp
                
        with col2:
            if st.button("❌ PASS", key=f"pass_{opp['position_key']}_{opp['strike']}"):
                record_decision(opp, "PASS", 0, opp['premium'], "Not attractive enough")
                st.success("Decision recorded!")
                st.rerun()

def record_decision(opp: Dict, decision: str, contracts: int, fill_price: float, reasoning: str):
    """Record trade decision"""
    trade_id = f"{opp['symbol']}_{opp['strike']}_{opp['expiration']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    st.session_state.decisions[trade_id] = {
        "decision": decision,
        "symbol": opp['symbol'],
        "strategy": "COVERED_CALL",
        "strike": opp['strike'],
        "expiration": opp['expiration'],
        "contracts": contracts,
        "quoted_premium": opp['premium'],
        "actual_fill": fill_price,
        "iv_rank": opp['iv_rank'],
        "growth_score": opp['growth_score'],
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
        "status": "ACTIVE" if decision == "TAKE" else "PASSED",
        "outcome": None,
        "profit_loss": None,
        "lessons_learned": None
    }
    
    save_json_data(DECISIONS_FILE, st.session_state.decisions)

def main():
    st.title("📈 COVERED CALL INCOME SYSTEM")
    st.markdown("**Mission:** Generate $2-5K Monthly Income + Pay Down $60K Margin Debt")
    
    initialize_session_state()
    
    alerts = check_21_50_7_alerts()
    if alerts:
        display_alerts(alerts)
    
    display_monthly_progress()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Opportunities", "📊 Positions", "🚀 Growth Scanner", "👁️ WatchList"])
    
    with tab1:
        if st.button("🔄 Scan for Opportunities"):
            with st.spinner("Scanning positions..."):
                opportunities = scan_covered_call_opportunities()
                st.session_state.opportunities = opportunities
        
        if hasattr(st.session_state, 'opportunities'):
            if st.session_state.opportunities:
                for opp in st.session_state.opportunities:
                    display_opportunity_card(opp)
                    st.markdown("---")
            else:
                st.info("No opportunities found. Check that your positions have liquid options.")
        else:
            st.info("Click 'Scan for Opportunities' to find covered call trades")
        
        if hasattr(st.session_state, 'show_take_dialog'):
            opp = st.session_state.show_take_dialog
            with st.form("take_form"):
                st.subheader(f"Execute Trade: {opp['symbol']} ${opp['strike']}")
                contracts = st.number_input(
                    "Contracts", 
                    min_value=1, 
                    max_value=opp['max_contracts'],
                    value=min(3, opp['max_contracts'])
                )
                fill_price = st.number_input(
                    "Fill Price", 
                    value=opp['premium'],
                    step=0.05
                )
                reasoning = st.text_area(
                    "Reasoning",
                    value="Post-earnings IV crush opportunity"
                )
                
                if st.form_submit_button("Confirm Trade"):
                    record_decision(opp, "TAKE", contracts, fill_price, reasoning)
                    del st.session_state.show_take_dialog
                    st.success("Trade executed!")
                    st.rerun()
    
    with tab2:
        add_position()
        st.markdown("---")
        display_positions()
    
    with tab3:
        st.subheader("🚀 Growth Stock Scanner")
        st.markdown("**Find the next NVDA, TSLA, or AMZN before they explode**")
        
        # Scanner controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            scan_type = st.selectbox(
                "Select Scan Type",
                [
                    "Early Innings: Next Giants ($1-50B)", 
                    "Tier 1: High Conviction (5-15 stocks)", 
                    "Tier 2: Expanded Screen (50-100 stocks)"
                ],
                help="Early Innings finds tomorrow's giants. Tier 1/2 find today's best."
            )
            if "Early Innings" in scan_type:
                tier = "early"
            else:
                tier = 1 if "Tier 1" in scan_type else 2
        
        with col2:
            universe = st.selectbox(
                "Universe",
                ["Top Growth Candidates", "S&P 500", "All US Stocks >$1B"],
                help="Start with Top Growth Candidates for testing"
            )
        
        with col3:
            if st.button("🔍 Run Scan", type="primary"):
                # Get symbols based on universe selection and tier
                if tier == "early":
                    # Early innings uses different universe
                    with st.spinner(f"Fetching early-stage stocks ($1B-$50B market cap)..."):
                        symbols = st.session_state.growth_screener.get_early_innings_candidates()
                        
                    with st.spinner(f"Scanning {len(symbols)} early-stage stocks..."):
                        results = []
                        for symbol in symbols:
                            result = st.session_state.growth_screener.screen_early_innings(symbol)
                            if result:
                                results.append(result)
                        
                        # Sort by early innings score
                        results.sort(key=lambda x: x.get('early_innings_score', 0), reverse=True)
                        st.session_state.scan_results = results
                else:
                    # Regular tier 1/2 scanning
                    if universe == "S&P 500":
                        symbols = st.session_state.growth_screener.get_sp500_symbols()
                    elif universe == "All US Stocks >$1B":
                        with st.spinner("Fetching all US stocks..."):
                            symbols = st.session_state.growth_screener.get_all_us_stocks()
                    else:  # Top Growth Candidates
                        symbols = st.session_state.growth_screener.get_sp500_symbols()
                    
                    with st.spinner(f"Scanning {len(symbols)} stocks..."):
                        results = st.session_state.growth_screener.screen_universe(symbols, tier=tier)
                        st.session_state.scan_results = results
        
        # Display results
        if hasattr(st.session_state, 'scan_results') and st.session_state.scan_results:
            results = st.session_state.scan_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stocks Found", len(results))
            with col2:
                # Handle both regular score and early_innings_score
                if results and results[0].get('tier') == 'early_innings':
                    high_conviction = len([r for r in results if r.get('early_innings_score', 0) >= 80])
                    st.metric("Future Giants (80+)", high_conviction)
                else:
                    high_conviction = len([r for r in results if r.get('score', 0) >= 80])
                    st.metric("High Conviction (80+)", high_conviction)
            with col3:
                # Handle both score types
                if results and results[0].get('tier') == 'early_innings':
                    avg_score = sum(r.get('early_innings_score', 0) for r in results) / len(results) if results else 0
                else:
                    avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
                st.metric("Avg Score", f"{avg_score:.0f}")
            with col4:
                st.metric("Scan Date", datetime.now().strftime("%Y-%m-%d"))
            
            st.markdown("---")
            
            # Results table
            for i, result in enumerate(results):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                    
                    with col1:
                        # Different display for early innings
                        if result.get('tier') == 'early_innings':
                            emoji = result.get('potential', '👀').split()[0]  # Get emoji from potential
                            st.markdown(f"### {emoji} {result['symbol']} - {result.get('potential', '')}")
                            st.write(f"**{result.get('industry', 'Unknown')}** | ${result.get('market_cap', 0)/1e9:.1f}B | Rev: ${result.get('revenue', 0)/1e6:.0f}M")
                        else:
                            # Regular tier 1/2 display
                            if result['score'] >= 85:
                                emoji = "🌟"  # Superstar
                            elif result['score'] >= 75:
                                emoji = "🎯"  # High conviction
                            elif result['score'] >= 65:
                                emoji = "💎"  # Hidden gem
                            else:
                                emoji = "👀"  # Watch list
                            
                            st.markdown(f"### {emoji} {result['symbol']}")
                            st.write(f"**{result.get('sector', 'Unknown')}** | Market Cap: ${result.get('market_cap', 0)/1e9:.1f}B")
                    
                    with col2:
                        if result.get('tier') == 'early_innings':
                            # Early innings uses different score
                            score = result.get('early_innings_score', 0)
                            score_color = "🚀" if score >= 80 else "💎" if score >= 65 else "🌱" if score >= 50 else "👀"
                            st.metric("Potential Score", f"{score_color} {score}/100")
                            
                            # Show conviction score
                            conv_score = result.get('conviction_score', 0)
                            conv_emoji = "🔥" if conv_score >= 85 else "🎯" if conv_score >= 70 else "💡" if conv_score >= 55 else "🤔"
                            st.metric("Conviction", f"{conv_emoji} {conv_score}/100")
                            st.write(f"**{result.get('conviction_level', '')}**")
                        else:
                            # Regular growth score
                            score_color = "🔴" if result['score'] >= 80 else "🟡" if result['score'] >= 60 else "🟢"
                            st.metric("Growth Score", f"{score_color} {result['score']}/100")
                            st.write(f"**{result['confidence']}**")
                    
                    with col3:
                        st.metric("Revenue Growth", f"{result.get('revenue_growth_yoy', 0)*100:.0f}%")
                        if result.get('tier') == 'early_innings':
                            st.write(f"**Insider: {result.get('insider_ownership', 0)*100:.0f}%**")
                        elif result.get('revenue_acceleration'):
                            st.write("📈 Accelerating!")
                    
                    with col4:
                        if result.get('tier') == 'early_innings':
                            st.metric("Gross Margin", f"{result.get('gross_margin', 0)*100:.0f}%")
                            st.metric("Employees", f"{result.get('employees', 0):,}")
                        else:
                            st.metric("PEG Ratio", f"{result.get('peg_ratio', 0):.2f}")
                            st.metric("Rel Strength", f"{result.get('relative_strength', 0):.2f}x")
                    
                    with col5:
                        if result['symbol'] in st.session_state.positions:
                            st.success("✓ Owned")
                            # Check appropriate score field
                            score = result.get('early_innings_score', result.get('score', 0))
                            if score >= 75:
                                st.warning("⚠️ NO CCs!")
                    
                    with col6:
                        # Load watchlist to check if already added
                        watchlist = load_json_data(WATCHLIST_FILE)
                        if result['symbol'] in watchlist:
                            st.info("👁️ Watching")
                        else:
                            if st.button(f"+ Watch", key=f"watch_{result['symbol']}"):
                                # Add to watchlist
                                watchlist[result['symbol']] = {
                                    "added_date": datetime.now().isoformat(),
                                    "added_price": 0,  # Will update on next price check
                                    "notes": f"From {result.get('tier', 'scan')} scan - Score: {result.get('early_innings_score', result.get('score', 0))}",
                                    "scan_data": result
                                }
                                save_json_data(WATCHLIST_FILE, watchlist)
                                st.rerun()
                    
                    # Expandable analysis
                    with st.expander(f"📊 Full Analysis for {result['symbol']}"):
                        if result.get('tier') == 'early_innings':
                            # Early innings analysis
                            st.write(f"**{result.get('recommendation', '')}**")
                            
                            # Motley Fool Style Analysis Section
                            st.markdown("---")
                            st.markdown("### 📰 Motley Fool Style Analysis")
                            
                            # What the company does
                            st.markdown("#### What This Company Does")
                            company_desc = result.get('company_description', 'No description available')
                            if len(company_desc) > 500:
                                st.write(company_desc[:500] + "...")
                            else:
                                st.write(company_desc)
                            
                            if result.get('website'):
                                st.write(f"🌐 Website: {result.get('website')}")
                            
                            # The Bull Case
                            st.markdown("#### 🐂 The Bull Case")
                            st.write("**Why This Could Be a 10-Bagger:**")
                            
                            # Growth story
                            if result.get('revenue_growth_yoy', 0) > 0.50:
                                st.write(f"• **Hypergrowth Story**: Revenue growing {result.get('revenue_growth_yoy', 0)*100:.0f}% YoY - this is NVDA 2016 territory!")
                            elif result.get('revenue_growth_yoy', 0) > 0.30:
                                st.write(f"• **Strong Growth**: {result.get('revenue_growth_yoy', 0)*100:.0f}% revenue growth puts it in elite company")
                            
                            # Market opportunity
                            if result.get('industry'):
                                st.write(f"• **Massive TAM**: Operating in {result.get('industry')} with huge expansion potential")
                            
                            # Quality metrics
                            if result.get('gross_margin', 0) > 0.70:
                                st.write(f"• **Software-Like Margins**: {result.get('gross_margin', 0)*100:.0f}% gross margins = pricing power")
                            
                            # Management
                            if result.get('insider_ownership', 0) > 0.15:
                                st.write(f"• **Founder-Led**: {result.get('insider_ownership', 0)*100:.0f}% insider ownership = aligned incentives")
                            
                            # Conviction factors
                            st.write("\n**Conviction Builders:**")
                            for factor in result.get('conviction_factors', []):
                                st.write(f"✅ {factor}")
                            
                            # The Bear Case
                            st.markdown("#### 🐻 The Bear Case")
                            st.write("**What Could Go Wrong:**")
                            
                            # Valuation
                            if result.get('ps_ratio', 0) > 10:
                                st.write(f"• **Rich Valuation**: P/S of {result.get('ps_ratio', 0):.1f}x requires flawless execution")
                            
                            # Competition
                            st.write("• **Competition Risk**: Larger players could enter the market")
                            
                            # Scale
                            if result.get('revenue', 0) < 500_000_000:
                                st.write(f"• **Execution Risk**: Still only ${result.get('revenue', 0)/1e6:.0f}M revenue - long way to go")
                            
                            # Market conditions
                            st.write("• **Market Risk**: Growth stocks can be volatile in downturns")
                            
                            # Investment recommendation
                            st.markdown("#### 💡 The Motley Fool Take")
                            
                            conv_score = result.get('conviction_score', 0)
                            if conv_score >= 85:
                                st.success("""
                                **BUY - This is a Rare Find**
                                
                                Companies with this combination of growth, margins, and market opportunity don't come along often. 
                                Start with a 2-3% position and add on any weakness. This could be a core holding for the next decade.
                                
                                *Recommended Action: Buy in thirds - 1/3 now, 1/3 on any 10% dip, 1/3 on confirmation of next quarter*
                                """)
                            elif conv_score >= 70:
                                st.info("""
                                **BUY - Strong Growth Story**
                                
                                The fundamentals are compelling, though not quite "back up the truck" territory. 
                                This deserves a spot in a growth portfolio, but size appropriately.
                                
                                *Recommended Action: Start with 1-2% position, add if thesis plays out*
                                """)
                            elif conv_score >= 55:
                                st.warning("""
                                **WATCH - Promising but Prove It**
                                
                                The pieces are there but we need to see more consistent execution. 
                                Add to watchlist and revisit after next earnings.
                                
                                *Recommended Action: Small starter position (0.5-1%) or wait for next quarter*
                                """)
                            else:
                                st.info("""
                                **PASS - Too Speculative**
                                
                                While interesting, there are likely better opportunities elsewhere. 
                                Keep on radar but don't chase.
                                
                                *Recommended Action: Watchlist only*
                                """)
                            
                            st.markdown("---")
                            
                            # Original metrics section
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("\n**Key Metrics:**")
                                st.write(f"• Market Cap: ${result.get('market_cap', 0)/1e9:.1f}B")
                                st.write(f"• Revenue: ${result.get('revenue', 0)/1e6:.0f}M")
                                st.write(f"• P/S Ratio: {result.get('ps_ratio', 0):.1f}x")
                                st.write(f"• QoQ Growth: {result.get('revenue_growth_qoq', 0)*100:.0f}%")
                            
                            with col2:
                                st.write("\n**Quality Indicators:**")
                                st.write(f"• Gross Margin: {result.get('gross_margin', 0)*100:.0f}%")
                                st.write(f"• Insider Own: {result.get('insider_ownership', 0)*100:.0f}%")
                                st.write(f"• Inst. Own: {result.get('institutional_ownership', 0)*100:.0f}%")
                                st.write(f"• 3M Momentum: {result.get('relative_strength', 0):.2f}x")
                        else:
                            # Regular analysis
                            st.write("**Growth Factors:**")
                            for factor in result.get('factors', []):
                                st.write(f"✅ {factor}")
                            
                            st.write("\n**Criteria Met:**")
                            criteria_cols = st.columns(2)
                            criteria_list = result.get('criteria_met', [])
                            for i, criteria in enumerate(criteria_list):
                                with criteria_cols[i % 2]:
                                    st.write(f"• {criteria.replace('_', ' ').title()}")
                            
                            st.write("\n**Covered Call Strategy:**")
                            score = result.get('score', 0)
                            if score >= 80:
                                st.error("🚫 DO NOT sell covered calls - protect the upside!")
                            elif score >= 70:
                                st.warning("⚠️ Only sell far OTM calls (15%+) or skip entirely")
                            elif score >= 60:
                                st.info("💡 Conservative covered calls OK (10-15% OTM)")
                            else:
                                st.success("✅ Standard covered call strategies appropriate")
                    
                    st.markdown("---")
            
            # Analysis and refinement section
            with st.expander("🧠 System Performance & Refinements"):
                analysis = st.session_state.growth_screener.analyze_and_refine()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Screening Effectiveness:**")
                    st.metric("Historical Success Rate", f"{analysis['success_rate']:.1%}")
                    st.write(f"Total Screens Analyzed: {analysis['total_screens']}")
                    
                    if analysis['best_criteria']:
                        st.write("\n**Most Predictive Criteria:**")
                        for criteria, rate in analysis['best_criteria'].items():
                            st.write(f"• {criteria}: {rate:.1%} success rate")
                
                with col2:
                    if analysis['recommended_adjustments']:
                        st.write("**Recommended Adjustments:**")
                        for adjustment in analysis['recommended_adjustments']:
                            st.info(f"💡 {adjustment}")
                    
                    st.write("\n**Next Steps:**")
                    st.write("1. System tracks all screening results")
                    st.write("2. Analyzes 30/90/180 day outcomes")
                    st.write("3. Automatically refines criteria")
                    st.write("4. Gets smarter over time!")
        
        else:
            st.info("👆 Click 'Run Scan' to find high-growth opportunities")
            
            # Educational content
            with st.expander("📚 How the Growth Scanner Works"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### 🚀 Early Innings Scanner
                    Finds tomorrow's giants ($1-50B market cap):
                    - 📈 Revenue growth >30% YoY minimum
                    - 💎 Gross margins >50%
                    - 💰 P/S ratio <20x (not overvalued)
                    - 👥 Insider ownership >10%
                    - 📊 Revenue >$100M (real business)
                    
                    **Potential Score (0-100):**
                    - 🚀 **80-100: FUTURE GIANT** - Next NVDA/SHOP
                    - 💎 **65-79: HIDDEN GEM** - High 10x potential
                    - 🌱 **50-64: EMERGING GROWTH** - Worth watching
                    - 👀 **0-49: EARLY STAGE** - Needs more proof
                    
                    **Conviction Score (0-100):**
                    - 🔥 **85+: EXTREME** - Back up the truck!
                    - 🎯 **70-84: HIGH** - Build significant position
                    - 💡 **55-69: MODERATE** - Start position
                    - 🤔 **0-54: SPECULATIVE** - Small position only
                    """)
                
                with col2:
                    st.markdown("""
                    ### 📊 Traditional Screens
                    
                    **Tier 1: High Conviction (5-15 stocks)**
                    - All criteria must be met
                    - Today's best growth stocks
                    - Higher market caps OK
                    
                    **Tier 2: Expanded (50-100 stocks)**
                    - 5+ criteria must be met
                    - Broader opportunity set
                    - Mix of growth profiles
                    
                    **Growth Score (0-100):**
                    - 🔴 **80-100:** No covered calls ever
                    - 🟡 **70-79:** Very limited CCs only
                    - 🟢 **60-69:** Conservative CCs OK
                    - ⚪ **0-59:** Standard CC strategies
                    
                    **Key Difference:**
                    Early Innings finds future 10-baggers.
                    Tier 1/2 find today's winners.
                    """)
                
                st.markdown("---")
                st.info("""
                💡 **Pro Tip:** Use Early Innings to find positions to buy and hold for 2-5+ years. 
                These are NOT covered call candidates - they're growth investments where you want unlimited upside!
                """)
            
            # Show recommendations based on existing positions
            st.markdown("### 💡 Recommendations for Your Positions")
            recommendations = st.session_state.growth_screener.get_recommended_positions()
            
            if recommendations['avoid_cc']:
                st.error(f"🚫 Never sell calls on: {', '.join(recommendations['avoid_cc'])}")
    
    with tab4:
        st.subheader("👁️ WatchList")
        st.markdown("**Track high-potential stocks before buying**")
        
        # Load watchlist
        watchlist = load_json_data(WATCHLIST_FILE)
        
        # Add to watchlist section
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_symbol = st.text_input("Add Symbol", placeholder="PLTR")
        with col2:
            notes = st.text_input("Notes", placeholder="AI play, waiting for pullback")
        with col3:
            if st.button("➕ Add to WatchList", type="primary"):
                if new_symbol:
                    symbol = new_symbol.upper()
                    # Get current price
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        # Try multiple price fields
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price', 0)
                        if current_price == 0:
                            hist = ticker.history(period="1d")
                            if not hist.empty:
                                current_price = hist['Close'].iloc[-1]
                        
                        watchlist[symbol] = {
                            "added_date": datetime.now().isoformat(),
                            "added_price": current_price,
                            "notes": notes,
                            "alerts": []
                        }
                        save_json_data(WATCHLIST_FILE, watchlist)
                        st.success(f"✅ Added {symbol} at ${current_price:.2f}")
                        st.rerun()
                    except:
                        st.error(f"Could not fetch data for {symbol}")
        
        # Display watchlist
        if watchlist:
            st.markdown("### Current WatchList")
            
            # Create DataFrame for display
            watchlist_data = []
            for symbol, data in watchlist.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    # Try multiple price fields as yfinance can be inconsistent
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price', 0)
                    # If still no price, try to get from recent history
                    if current_price == 0:
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                    added_price = data.get('added_price', 0)
                    
                    # Calculate performance
                    change_pct = ((current_price - added_price) / added_price * 100) if added_price > 0 else 0
                    
                    # Check if it's from growth scanner
                    if hasattr(st.session_state, 'scan_results'):
                        scan_result = next((r for r in st.session_state.scan_results if r['symbol'] == symbol), None)
                        score = scan_result.get('early_innings_score', scan_result.get('score', 0)) if scan_result else 0
                    else:
                        score = 0
                    
                    watchlist_data.append({
                        "Symbol": symbol,
                        "Added": data['added_date'][:10],
                        "Added Price": f"${added_price:.2f}",
                        "Current": f"${current_price:.2f}",
                        "Change": f"{change_pct:+.1f}%",
                        "Score": score if score > 0 else "-",
                        "Notes": data.get('notes', '')
                    })
                except:
                    pass
            
            if watchlist_data:
                df = pd.DataFrame(watchlist_data)
                
                # Display with actions
                for idx, row in df.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5, 1, 1, 1, 1, 1, 3, 1])
                        
                        with col1:
                            st.markdown(f"**{row['Symbol']}**")
                        with col2:
                            st.text(row['Added'])
                        with col3:
                            st.text(row['Added Price'])
                        with col4:
                            st.text(row['Current'])
                        with col5:
                            # Color code the change
                            change_val = float(row['Change'].replace('%', '').replace('+', ''))
                            if change_val > 0:
                                st.markdown(f"🟢 {row['Change']}")
                            elif change_val < 0:
                                st.markdown(f"🔴 {row['Change']}")
                            else:
                                st.text(row['Change'])
                        with col6:
                            if row['Score'] != "-":
                                st.metric("Score", row['Score'])
                            else:
                                st.text("-")
                        with col7:
                            st.text(row['Notes'])
                        with col8:
                            if st.button("🗑️", key=f"remove_{row['Symbol']}"):
                                del watchlist[row['Symbol']]
                                save_json_data(WATCHLIST_FILE, watchlist)
                                st.rerun()
                
                st.markdown("---")
                
                # Quick actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Move Best to Positions"):
                        # Find top scoring watchlist items
                        best_symbols = [row['Symbol'] for row in watchlist_data if row['Score'] != "-" and float(row['Score']) >= 80]
                        if best_symbols:
                            st.info(f"Ready to add: {', '.join(best_symbols[:3])}")
                            st.write("Go to Positions tab to add these high-conviction plays!")
                        else:
                            st.warning("No high-conviction (80+) stocks in watchlist yet")
                
                with col2:
                    if st.button("🔄 Update Scores"):
                        with st.spinner("Analyzing watchlist..."):
                            for symbol in watchlist.keys():
                                # Run early innings scan on each
                                result = st.session_state.growth_screener.screen_early_innings(symbol)
                                if result:
                                    watchlist[symbol]['latest_score'] = result.get('early_innings_score', 0)
                                    watchlist[symbol]['conviction'] = result.get('conviction_score', 0)
                            save_json_data(WATCHLIST_FILE, watchlist)
                            st.success("✅ Scores updated!")
                            st.rerun()
        else:
            st.info("👀 No stocks in watchlist yet. Add some from the Growth Scanner results!")

if __name__ == "__main__":
    main()