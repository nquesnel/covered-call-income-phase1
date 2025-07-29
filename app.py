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

def load_json_data(filepath: str, default=None):
    """Load data from JSON file, return default if file doesn't exist"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    # Return default value if provided, otherwise empty dict
    return default if default is not None else {}

def save_json_data(filepath: str, data) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_demo_portfolio():
    """Load Neal's complete portfolio"""
    positions = [
        # Traditional IRA
        {"symbol": "AMZN", "shares": 60, "cost_basis": 37.83, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 85},
        {"symbol": "BRK.B", "shares": 20, "cost_basis": 124.44, "account_type": "roth", "growth_category": "Conservative", "growth_score": 25},
        {"symbol": "COIN", "shares": 20, "cost_basis": 228.75, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 72},
        {"symbol": "CRWD", "shares": 10, "cost_basis": 206.91, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 88},
        {"symbol": "HIMS", "shares": 100, "cost_basis": 28.06, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 82},
        {"symbol": "MA", "shares": 52.069, "cost_basis": 86.67, "account_type": "roth", "growth_category": "Moderate", "growth_score": 48},
        {"symbol": "RKLB", "shares": 75, "cost_basis": 28.24, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 90},
        {"symbol": "TDOC", "shares": 75, "cost_basis": 105.47, "account_type": "roth", "growth_category": "Conservative", "growth_score": 20},
        {"symbol": "TSLA", "shares": 5, "cost_basis": 271.94, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 75},
        {"symbol": "TWLO", "shares": 50, "cost_basis": 29.79, "account_type": "roth", "growth_category": "Conservative", "growth_score": 30},
        # Taxable Account 1
        {"symbol": "AAPL", "shares": 244.9777, "cost_basis": 43.99, "account_type": "taxable", "growth_category": "Moderate", "growth_score": 48},
        {"symbol": "ADBE", "shares": 10, "cost_basis": 29.45, "account_type": "taxable", "growth_category": "Moderate", "growth_score": 55},
        {"symbol": "COST", "shares": 25.079, "cost_basis": 78.46, "account_type": "taxable", "growth_category": "Moderate", "growth_score": 38},
        {"symbol": "EXEL", "shares": 200, "cost_basis": 4.97, "account_type": "taxable", "growth_category": "Conservative", "growth_score": 20},
        {"symbol": "GOOGL", "shares": 60.351, "cost_basis": 12.95, "account_type": "taxable", "growth_category": "Moderate", "growth_score": 58},
        {"symbol": "BAC", "shares": 200, "cost_basis": 5.49, "account_type": "taxable", "growth_category": "Conservative", "growth_score": 18},
        {"symbol": "COIN", "shares": 15, "cost_basis": 193.75, "account_type": "taxable", "growth_category": "Aggressive", "growth_score": 72},
        # TSLA taxable
        {"symbol": "TSLA", "shares": 15, "cost_basis": 251.40, "account_type": "taxable", "growth_category": "Aggressive", "growth_score": 75},
        # Roth IRA
        {"symbol": "BB", "shares": 500, "cost_basis": 4.28, "account_type": "roth", "growth_category": "Conservative", "growth_score": 15},
        {"symbol": "CRWD", "shares": 7, "cost_basis": 319.50, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 88},
        {"symbol": "GENI", "shares": 300, "cost_basis": 4.33, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 65},
        {"symbol": "GXO", "shares": 100, "cost_basis": 0.00, "account_type": "roth", "growth_category": "Moderate", "growth_score": 40},
        {"symbol": "HIMS", "shares": 30, "cost_basis": 41.67, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 82},
        {"symbol": "HUBS", "shares": 20, "cost_basis": 115.38, "account_type": "roth", "growth_category": "Moderate", "growth_score": 55},
        {"symbol": "HWM", "shares": 10, "cost_basis": 153.57, "account_type": "roth", "growth_category": "Moderate", "growth_score": 45},
        {"symbol": "MDB", "shares": 100, "cost_basis": 56.60, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 82},
        {"symbol": "MGNI", "shares": 500, "cost_basis": 6.22, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 68},
        {"symbol": "OKTA", "shares": 100, "cost_basis": 29.82, "account_type": "roth", "growth_category": "Moderate", "growth_score": 52},
        {"symbol": "OSCR", "shares": 150, "cost_basis": 14.27, "account_type": "roth", "growth_category": "Hypergrowth", "growth_score": 85},
        {"symbol": "PLTR", "shares": 100, "cost_basis": 40.35, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 78},
        {"symbol": "RXO", "shares": 100, "cost_basis": 0.00, "account_type": "roth", "growth_category": "Moderate", "growth_score": 40},
        {"symbol": "TSLA", "shares": 15, "cost_basis": 296.10, "account_type": "roth", "growth_category": "Aggressive", "growth_score": 75},
        {"symbol": "TWLO", "shares": 35, "cost_basis": 106.19, "account_type": "roth", "growth_category": "Conservative", "growth_score": 30},
        {"symbol": "XPO", "shares": 100, "cost_basis": 27.97, "account_type": "roth", "growth_category": "Moderate", "growth_score": 42}
    ]
    
    # Add each position with growth scoring
    for pos in positions:
        # Skip if position already exists
        existing = [p for p in st.session_state.positions if isinstance(p, dict) and p.get('symbol') == pos['symbol'] and p.get('account_type') == pos['account_type']]
        if not existing:
            # Use pre-calculated scores if available, otherwise calculate
            if 'growth_score' not in pos or 'growth_category' not in pos:
                score, category, _ = calculate_growth_score(pos['symbol'])
                pos['growth_category'] = category
                pos['growth_score'] = score
            pos['added_date'] = datetime.now().isoformat()
            st.session_state.positions.append(pos)
    
    # Save to file
    save_json_data(POSITIONS_FILE, st.session_state.positions)

def initialize_session_state():
    """Initialize session state variables"""
    if 'positions' not in st.session_state:
        data = load_json_data(POSITIONS_FILE)
        # Ensure positions is always a list
        if isinstance(data, dict):
            st.session_state.positions = []
        else:
            st.session_state.positions = data if data else []
    if 'decisions' not in st.session_state:
        st.session_state.decisions = load_json_data(DECISIONS_FILE)
    if 'growth_screener' not in st.session_state:
        st.session_state.growth_screener = GrowthScreeningSystem()
    if 'active_options' not in st.session_state:
        st.session_state.active_options = load_json_data(os.path.join(DATA_DIR, "active_options.json"), [])

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
    income_goal = 3500
    progress = min(monthly_income / income_goal, 1.0) if income_goal > 0 else 0
    
    # Simple Streamlit metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 INCOME TARGET", f"${income_goal:,}", f"${monthly_income:,.0f}")
    with col2:
        st.metric("🎯 WIN RATE", f"{win_rate:.1f}%")
    with col3:
        st.metric("⚡ ACTIVE TRADES", active_trades)
    with col4:
        st.metric("🎲 MARGIN DEBT", "$60,000", "-$2,800")

def add_option_trade():
    """Add active option trade"""
    with st.expander("🎯 Add Option Trade", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="TSLA")
            option_type = st.selectbox("Type", ["COVERED CALL", "CASH SECURED PUT", "LONG CALL", "LONG PUT"])
            strike = st.number_input("Strike Price", min_value=0.0, value=100.0, step=1.0)
        
        with col2:
            expiration = st.date_input("Expiration Date")
            contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
            premium = st.number_input("Premium Collected/Paid", value=1.0, step=0.01)
        
        with col3:
            fill_price = st.number_input("Fill Price", value=1.0, step=0.01)
            account = st.selectbox("Account", ["taxable", "roth"])
            notes = st.text_area("Notes", placeholder="Earnings play, high IV...")
        
        if st.button("➕ Add Option Trade", type="primary"):
            if symbol and strike > 0:
                option_trade = {
                    "id": f"{symbol}_{strike}_{expiration}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "symbol": symbol.upper(),
                    "type": option_type,
                    "strike": strike,
                    "expiration": expiration.strftime("%Y-%m-%d"),
                    "contracts": contracts,
                    "premium": premium,
                    "fill_price": fill_price,
                    "account": account,
                    "notes": notes,
                    "opened_date": datetime.now().strftime("%Y-%m-%d"),
                    "status": "ACTIVE",
                    "pnl": 0
                }
                
                st.session_state.active_options.append(option_trade)
                save_json_data(os.path.join(DATA_DIR, "active_options.json"), st.session_state.active_options)
                st.success(f"Added {option_type} for {symbol}")
                st.rerun()

def display_active_options():
    """Display active options positions"""
    if not st.session_state.active_options:
        st.info("No active options positions. Track your covered calls here!")
        return
    
    # Summary metrics
    active_calls = [o for o in st.session_state.active_options if "CALL" in o['type'] and o['status'] == "ACTIVE"]
    active_puts = [o for o in st.session_state.active_options if "PUT" in o['type'] and o['status'] == "ACTIVE"]
    total_premium = sum(o['premium'] * o['contracts'] * 100 for o in st.session_state.active_options if o['status'] == "ACTIVE")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Calls", len(active_calls))
    with col2:
        st.metric("Active Puts", len(active_puts))
    with col3:
        st.metric("Total Premium", f"${total_premium:,.2f}")
    with col4:
        st.metric("At Risk", len([o for o in st.session_state.active_options if o['status'] == "ACTIVE"]))
    
    # Active positions table
    for option in st.session_state.active_options:
        if option['status'] == "ACTIVE":
            exp_date = datetime.strptime(option['expiration'], "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days
            
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{option['symbol']}** {option['type']}")
            with col2:
                st.write(f"${option['strike']}")
            with col3:
                st.write(f"{option['expiration']} ({days_to_exp}d)")
            with col4:
                st.write(f"{option['contracts']} x ${option['premium']:.2f}")
            with col5:
                if days_to_exp <= 7:
                    st.write("🔴 EXPIRING")
                elif days_to_exp <= 21:
                    st.write("🟡 MANAGE")
                else:
                    st.write("🟢 ACTIVE")
            with col6:
                if st.button("Close", key=f"close_{option['id']}"):
                    option['status'] = "CLOSED"
                    option['closed_date'] = datetime.now().strftime("%Y-%m-%d")
                    save_json_data(os.path.join(DATA_DIR, "active_options.json"), st.session_state.active_options)
                    st.rerun()

def bulk_import_positions():
    """Bulk import positions from CSV file"""
    with st.expander("📋 Bulk Import Positions", expanded=False):
        # Simple CSV file uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv', 'txt'])
        
        st.markdown("""
        **Simple format (4 columns):**
        ```
        AAPL,100,150.00,taxable
        MSFT,200,300.00,roth
        ```
        """)
        
        if uploaded_file is not None:
            try:
                # Read the file
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                imported = 0
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 4:
                            symbol = parts[0].strip().upper()
                            shares = float(parts[1].strip())
                            cost_basis = float(parts[2].strip())
                            account_type = parts[3].strip().lower()
                            
                            # Get growth score
                            score, category, _ = calculate_growth_score(symbol)
                            
                            # Add position
                            position = {
                                "symbol": symbol,
                                "shares": shares,
                                "cost_basis": cost_basis,
                                "account_type": account_type,
                                "growth_category": category,
                                "growth_score": score,
                                "added_date": datetime.now().isoformat()
                            }
                            
                            # Check if position already exists
                            exists = any(p['symbol'] == symbol and p['account_type'] == account_type 
                                       for p in st.session_state.positions)
                            if not exists:
                                st.session_state.positions.append(position)
                                imported += 1
                
                if imported > 0:
                    save_json_data(POSITIONS_FILE, st.session_state.positions)
                    st.success(f"✅ Imported {imported} positions!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def add_position():
    """Add new position form"""
    with st.expander("➕ Add New Position", expanded=False):
        add_position_form()

def add_position_form():
    """The actual form for adding positions"""
    
    # Quick load demo portfolio
    if st.button("🚀 Load My Portfolio (One Click)", type="primary", help="Load Neal's complete portfolio"):
        load_demo_portfolio()
        st.success("✅ Loaded 34 positions!")
        st.rerun()
    
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
    """Display premium portfolio command center"""
    
    # Portfolio Summary Header
    portfolio_data = display_portfolio_summary()
    
    # Position Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📊 PORTFOLIO POSITIONS")
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🧹 Remove Dupes", type="secondary"):
                # Remove duplicates
                seen = set()
                unique_positions = []
                for pos in st.session_state.positions:
                    key = (pos['symbol'], pos['account_type'])
                    if key not in seen:
                        seen.add(key)
                        unique_positions.append(pos)
                
                removed = len(st.session_state.positions) - len(unique_positions)
                st.session_state.positions = unique_positions
                save_json_data(POSITIONS_FILE, st.session_state.positions)
                if removed > 0:
                    st.success(f"Removed {removed} duplicates!")
                    st.rerun()
                    
        with col2b:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.positions = []
                save_json_data(POSITIONS_FILE, st.session_state.positions)
                st.rerun()
    
    if not st.session_state.positions:
        st.info("No positions yet. Add your first position above!")
        return
    
    # Display positions as premium trading cards
    # Create 3-column layout for cards
    positions = st.session_state.positions
    cols_per_row = 3
    
    for i in range(0, len(positions), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(positions):
                with cols[j]:
                    display_position_card(positions[i + j], i + j, portfolio_data['total_value'])

def display_portfolio_summary():
    """Display premium portfolio summary header"""
    # Calculate portfolio metrics
    total_value = 0
    total_cost = 0
    total_pnl = 0
    best_performer = None
    worst_performer = None
    best_pnl_pct = -float('inf')
    worst_pnl_pct = float('inf')
    
    for pos in st.session_state.positions:
        try:
            ticker = yf.Ticker(pos['symbol'])
            current_price = ticker.info.get('currentPrice', pos['cost_basis'])
            position_value = current_price * pos['shares']
            position_cost = pos['cost_basis'] * pos['shares']
            position_pnl = position_value - position_cost
            pnl_pct = (position_pnl / position_cost * 100) if position_cost > 0 else 0
            
            total_value += position_value
            total_cost += position_cost
            total_pnl += position_pnl
            
            if pnl_pct > best_pnl_pct:
                best_pnl_pct = pnl_pct
                best_performer = pos['symbol']
            if pnl_pct < worst_pnl_pct:
                worst_pnl_pct = pnl_pct
                worst_performer = pos['symbol']
        except:
            pass
    
    # Simple Streamlit metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💼 PORTFOLIO VALUE", f"${total_value:,.2f}", f"${total_pnl:+,.2f}")
        
    with col2:
        pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        st.metric("📈 TOTAL RETURN", f"{pnl_pct:+.2f}%", f"${total_pnl:+,.2f}")
        
    with col3:
        if best_performer:
            st.metric("🚀 BEST PERFORMER", best_performer, f"{best_pnl_pct:+.1f}%")
        else:
            st.metric("🚀 BEST PERFORMER", "—", "—")
            
    with col4:
        if worst_performer:
            st.metric("📉 WORST PERFORMER", worst_performer, f"{worst_pnl_pct:+.1f}%")
        else:
            st.metric("📉 WORST PERFORMER", "—", "—")
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'best_performer': best_performer,
        'worst_performer': worst_performer
    }

def display_position_card(position, idx, total_portfolio_value):
    """Display a single position as a clean card with all content inside the box and icon actions in the top right."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(position['symbol'])
        info = ticker.info
        current_price = info.get('currentPrice', position['cost_basis'])
    except Exception:
        current_price = position['cost_basis']
    
    position_value = current_price * position['shares']
    position_cost = position['cost_basis'] * position['shares']
    position_pnl = position_value - position_cost
    pnl_pct = (position_pnl / position_cost * 100) if position_cost > 0 else 0
    portfolio_pct = (position_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
    growth_score = position.get('growth_score', 50)
    growth_category = position.get('growth_category', 'MODERATE')
    growth_color = '#00D2FF' if growth_score < 40 else '#FF6B35' if growth_score < 70 else '#FF073A'

    # Card CSS
    st.markdown(f"""
    <style>
    .position-card-{idx} {{
        background: #1A1A1A;
        border: 2px solid #444444;
        border-radius: 12px;
        padding: 24px 20px 32px 20px;
        margin: 16px 0;
        position: relative;
        min-height: 270px;
        box-sizing: border-box;
        color: #fff;
    }}
    .card-actions-{idx} {{
        position: absolute;
        top: 2px;
        right: 2px;
        display: flex;
        gap: 2px;
        z-index: 2;
    }}
    .card-btn-{idx} {{
        background: #000000;
        border: none;
        border-radius: 6px;
        padding: 4px 6px;
        margin: 0;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        transition: background 0.2s;
    }}
    .card-btn-{idx}:hover {{
        background: #222;
    }}
    .growth-meter-{idx} {{
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 10px;
        background: #222;
        border-radius: 0 0 10px 10px;
        border-top: 1px solid #444444;
        overflow: hidden;
    }}
    .growth-bar-{idx} {{
        height: 100%;
        background: {growth_color};
        width: {growth_score}%;
        transition: width 0.5s;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Card content as a single HTML string
    card_html = f'''
    <div class="position-card-{idx}">
        <div style="padding-right: 60px;">
            <h4 style="margin-bottom:0.5em">{position['symbol']} ({position['account_type'].upper()})</h4>
            <div><b>Shares:</b> {position['shares']}</div>
            <div><b>Cost Basis:</b> ${position['cost_basis']:.2f}</div>
            <div><b>Current Price:</b> ${current_price:.2f}</div>
            <div><b>Position Value:</b> ${position_value:,.2f}</div>
            <div><b>P&amp;L:</b> ${position_pnl:+,.2f} ({pnl_pct:+.1f}%)</div>
            <div><b>Portfolio %:</b> {portfolio_pct:.2f}%</div>
            <div><b>Growth Category:</b> {growth_category}</div>
            <div><b>Growth Score:</b> {growth_score}/100</div>
        </div>
        <div class="growth-meter-{idx}"><div class="growth-bar-{idx}"></div></div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)

    # Action buttons absolutely positioned (Streamlit-native, but visually inside the card)
    col1, col2, _ = st.columns([0.06, 0.06, 0.88])
    with col1:
        if st.button("✏️", key=f"edit_{idx}", help="Edit"):
            st.session_state[f'editing_{idx}'] = True
            st.rerun()
    with col2:
        if st.button("🗑️", key=f"del_{idx}", help="Delete"):
            del st.session_state.positions[idx]
            save_json_data(POSITIONS_FILE, st.session_state.positions)
            st.rerun()

    # Edit form (if editing)
    if st.session_state.get(f'editing_{idx}', False):
        with st.expander("Edit Position", expanded=True):
            new_shares = st.number_input("Shares", value=float(position['shares']), key=f"shares_{idx}")
            new_cost = st.number_input("Cost Basis", value=float(position['cost_basis']), key=f"cost_{idx}")
            new_account = st.selectbox("Account", ["taxable", "roth"], index=0 if position['account_type'] == "taxable" else 1, key=f"account_{idx}")
            if st.button("💾 Save", key=f"save_{idx}"):
                st.session_state.positions[idx]['shares'] = new_shares
                st.session_state.positions[idx]['cost_basis'] = new_cost
                st.session_state.positions[idx]['account_type'] = new_account
                score, category, _ = calculate_growth_score(position['symbol'])
                st.session_state.positions[idx]['growth_category'] = category
                st.session_state.positions[idx]['growth_score'] = score
                save_json_data(POSITIONS_FILE, st.session_state.positions)
                st.session_state[f'editing_{idx}'] = False
                st.success("Position updated!")
                st.rerun()
            if st.button("❌ Cancel", key=f"cancel_{idx}"):
                st.session_state[f'editing_{idx}'] = False
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

def calculate_greeks_score(greeks):
    """Calculate a score based on Greeks favorability for covered calls"""
    score = 5  # Start neutral
    
    # Delta scoring (lower is better for covered calls)
    delta = abs(greeks.get('delta', 0))
    if delta < 0.2:
        score += 2  # Very low assignment risk
    elif delta < 0.35:
        score += 1  # Low assignment risk
    elif delta > 0.5:
        score -= 2  # High assignment risk
    elif delta > 0.4:
        score -= 1  # Moderate assignment risk
    
    # Theta scoring (higher is better - more decay)
    theta = abs(greeks.get('theta', 0))
    if theta > 0.05:
        score += 2  # Excellent decay
    elif theta > 0.02:
        score += 1  # Good decay
    elif theta < 0.01:
        score -= 1  # Poor decay
    
    # Gamma scoring (lower is better - less risk)
    gamma = greeks.get('gamma', 0)
    if gamma < 0.02:
        score += 1  # Low gamma risk
    elif gamma > 0.05:
        score -= 1  # High gamma risk
    
    return max(0, min(10, score))  # Keep between 0-10

def analyze_covered_call_opportunity(position, call, current_price, iv_rank, growth_score, strike_multiplier, monthly_yield):
    """Analyze covered call opportunity and provide Motley Fool style recommendation"""
    
    # Calculate key metrics
    upside_to_strike = ((call['strike'] - current_price) / current_price) * 100
    breakeven_price = current_price - call['ask']
    downside_protection = ((current_price - breakeven_price) / current_price) * 100
    
    # Greeks analysis
    delta = abs(call.get('delta', 0))
    theta = abs(call.get('theta', 0))
    
    # Initialize recommendation
    rec = {
        'verdict': '',
        'reasoning': '',
        'action': '',
        'conditional': ''
    }
    
    # High growth stocks (score > 70) need more upside room
    if growth_score > 70:
        if strike_multiplier >= 1.10:  # 10%+ upside
            if monthly_yield > 1.5:
                rec['verdict'] = "🟢 STRONG YES"
                rec['reasoning'] = f"Excellent balance! {upside_to_strike:.1f}% upside + {monthly_yield:.1f}% monthly income on high-growth stock"
                rec['action'] = "Sell calls on 50-75% of position"
            elif monthly_yield > 0.8:
                rec['verdict'] = "🟡 YES with conditions"
                rec['reasoning'] = f"Good upside room ({upside_to_strike:.1f}%) but modest yield ({monthly_yield:.1f}%)"
                rec['conditional'] = f"Wait for IV spike above {iv_rank + 20:.0f}% or stock rally to ${current_price * 1.03:.2f}"
            else:
                rec['verdict'] = "🔴 NO - Better opportunities exist"
                rec['reasoning'] = "Premium too low for the risk on growth stock"
                rec['conditional'] = f"Revisit if premium reaches ${call['ask'] * 1.5:.2f}"
        else:
            rec['verdict'] = "🔴 NO - Caps growth too much"
            rec['reasoning'] = f"Only {upside_to_strike:.1f}% upside on high-growth stock"
            rec['conditional'] = "Look for strikes 10%+ OTM instead"
    
    # Moderate growth stocks (score 40-70)
    elif growth_score >= 40:
        if monthly_yield > 2.0:
            rec['verdict'] = "🟢 YES - Great income"
            rec['reasoning'] = f"{monthly_yield:.1f}% monthly yield with {upside_to_strike:.1f}% upside"
            rec['action'] = "Sell calls on 75-100% of position"
        elif monthly_yield > 1.0 and upside_to_strike > 5:
            rec['verdict'] = "🟡 MAYBE - Decent setup"
            rec['reasoning'] = f"Balanced {monthly_yield:.1f}% yield + {upside_to_strike:.1f}% upside"
            rec['conditional'] = f"YES if stock drops to ${current_price * 0.97:.2f} (better entry)"
        else:
            rec['verdict'] = "🟡 WAIT for better premium"
            rec['reasoning'] = "Current yield doesn't justify the cap risk"
            rec['conditional'] = f"Target {monthly_yield * 1.5:.1f}% monthly yield"
    
    # Conservative/value stocks (score < 40)
    else:
        if monthly_yield > 1.5:
            rec['verdict'] = "🟢 YES - Income focus"
            rec['reasoning'] = f"Strong {monthly_yield:.1f}% yield on stable stock"
            rec['action'] = "Sell calls on 100% of position"
        elif iv_rank > 70:
            rec['verdict'] = "🟡 YES - High volatility play"
            rec['reasoning'] = f"IV Rank {iv_rank:.0f}% suggests inflated premiums"
            rec['action'] = "Sell now, buy back on IV crush"
        else:
            rec['verdict'] = "🔴 NO - Low reward"
            rec['reasoning'] = "Better to hold or sell stock"
            rec['conditional'] = "Consider selling the stock instead"
    
    # Add IV-based adjustments
    if iv_rank > 80:
        rec['reasoning'] += f" | ⚡ IV Rank {iv_rank:.0f}% = PREMIUM ALERT!"
    elif iv_rank < 30:
        rec['reasoning'] += f" | 💤 IV Rank {iv_rank:.0f}% = Wait for volatility"
    
    # Add Greeks-based adjustments
    if delta > 0:
        if delta < 0.2:
            rec['reasoning'] += f" | 🎯 Delta {delta:.2f} = Very low assignment risk"
        elif delta > 0.5:
            rec['reasoning'] += f" | ⚠️ Delta {delta:.2f} = High assignment risk"
            if "YES" in rec['verdict'] and "STRONG" not in rec['verdict']:
                rec['verdict'] = rec['verdict'].replace("YES", "MAYBE")
                rec['conditional'] = "Consider rolling if stock rallies early"
    
    if theta > 0:
        daily_decay = theta * 100  # Convert to dollars
        if daily_decay > 5:
            rec['reasoning'] += f" | 💰 Theta ${daily_decay:.2f}/day = Excellent decay"
        elif daily_decay < 1:
            rec['reasoning'] += f" | 🐌 Theta ${daily_decay:.2f}/day = Slow decay"
    
    return rec

def scan_covered_call_opportunities():
    """Scan positions for covered call opportunities using user-defined filters"""
    opportunities = []
    
    # Get user-defined filters or use defaults
    filters = st.session_state.get('scan_filters', {
        'min_yield_growth': 0.5,
        'min_yield_moderate': 0.8,
        'min_yield_conservative': 1.0,
        'min_iv_rank': 20,
        'max_spread': 0.75,
        'min_premium_pct': 0.003,
        'include_earnings': False,
        'show_all_recommendations': True,
        'days_range': (21, 35)
    })
    
    for position in st.session_state.positions:
        # Skip positions with less than 100 shares
        if position['shares'] < 100:
            continue
        try:
            ticker = yf.Ticker(position['symbol'])
            current_price = ticker.info.get('currentPrice', 0)
            
            if current_price == 0:
                continue
            
            options_dates = ticker.options
            if not options_dates:
                continue
            
            # Use user-defined date range
            min_days, max_days = filters['days_range']
            target_dates = [
                datetime.now() + timedelta(days=min_days),
                datetime.now() + timedelta(days=(min_days + max_days) // 2),
                datetime.now() + timedelta(days=max_days)
            ]
            
            for target_date in target_dates:
                best_date = min(options_dates, 
                              key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
                
                opt_chain = ticker.option_chain(best_date)
                calls = opt_chain.calls
                
                # TIGHTER STRIKES: Based on growth category
                growth_category = position.get('growth_category', 'MODERATE')
                
                if growth_category in ['Hypergrowth', 'Aggressive']:
                    # High growth: Only 10-15% OTM
                    strikes_to_analyze = [
                        (1.10, "10% OTM - Balanced growth/income"),
                        (1.12, "12% OTM - Growth-friendly"),
                        (1.15, "15% OTM - Maximum upside")
                    ]
                elif growth_category == 'Moderate':
                    # Moderate growth: 5-10% OTM
                    strikes_to_analyze = [
                        (1.05, "5% OTM - Income focused"),
                        (1.08, "8% OTM - Balanced approach"),
                        (1.10, "10% OTM - Growth tilt")
                    ]
                else:  # Conservative
                    # Value stocks: 3-7% OTM
                    strikes_to_analyze = [
                        (1.03, "3% OTM - High income"),
                        (1.05, "5% OTM - Standard play"),
                        (1.07, "7% OTM - Conservative growth")
                    ]
                
                for strike_multiplier, strike_desc in strikes_to_analyze:
                    target_strike = current_price * strike_multiplier
                    
                    # Find closest strike to our target
                    if len(calls) > 0:
                        closest_strike_idx = (calls['strike'] - target_strike).abs().idxmin()
                        call = calls.loc[closest_strike_idx]
                        
                        if call['bid'] > 0:  # Only if there's a real bid
                            iv = call.get('impliedVolatility', 0.3)
                            iv_rank = calculate_iv_rank(position['symbol'], iv)
                            
                            growth_score = get_growth_score(
                                position['symbol'], 
                                position['growth_category']
                            )
                            
                            premium = (call['bid'] + call['ask']) / 2
                            days_to_exp = (datetime.strptime(best_date, '%Y-%m-%d') - datetime.now()).days
                            monthly_yield = (premium / current_price) * 100 * (30 / days_to_exp)
                            annual_yield = monthly_yield * 12
                            
                            # QUALITY FILTERS - Using user-defined settings
                            # 1. Minimum premium requirements
                            min_premium = current_price * filters['min_premium_pct']
                            if premium < min_premium:
                                continue
                                
                            # 2. Minimum monthly yield based on category
                            if growth_category in ['Hypergrowth', 'Aggressive']:
                                min_yield = filters['min_yield_growth']
                            elif growth_category == 'Moderate':
                                min_yield = filters['min_yield_moderate']
                            else:
                                min_yield = filters['min_yield_conservative']
                                
                            if monthly_yield < min_yield:
                                continue
                                
                            # 3. IV Rank filter
                            if iv_rank < filters['min_iv_rank']:
                                continue
                                
                            # 4. Bid-Ask spread filter
                            spread = (call['ask'] - call['bid']) / call['bid'] if call['bid'] > 0 else 1
                            if spread > filters['max_spread']:
                                continue
                            
                            # Check for upcoming earnings
                            earnings_date = None
                            try:
                                calendar = ticker.calendar
                                if calendar is not None and not calendar.empty:
                                    next_earnings = calendar.iloc[0]
                                    earnings_date = next_earnings.get('Earnings Date')
                                    if earnings_date:
                                        earnings_date = pd.to_datetime(earnings_date).strftime('%Y-%m-%d')
                            except:
                                pass
                            
                            # Calculate days to earnings if available
                            days_to_earnings = None
                            if earnings_date:
                                earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
                                days_to_earnings = (earnings_dt - datetime.now()).days
                            
                            # Calculate recommendation based on multiple factors
                            recommendation = analyze_covered_call_opportunity(
                                position, call, current_price, iv_rank, growth_score, 
                                strike_multiplier, monthly_yield
                            )
                            
                            # Adjust recommendation based on earnings proximity
                            if days_to_earnings and days_to_earnings <= 14:
                                if not filters['include_earnings']:
                                    # Skip opportunities too close to earnings
                                    continue
                                else:
                                    # Include but warn strongly
                                    recommendation['verdict'] = "🔴 EARNINGS WARNING"
                                    recommendation['reasoning'] = f"⚠️ EARNINGS in {days_to_earnings} days! High risk!"
                            elif days_to_earnings and days_to_earnings <= 30:
                                # Add earnings warning to existing recommendation
                                if "YES" in recommendation['verdict']:
                                    recommendation['verdict'] = recommendation['verdict'].replace("YES", "MAYBE")
                                recommendation['reasoning'] += f" | ⚠️ Earnings {earnings_date} ({days_to_earnings} days)"
                            
                            # Filter recommendations based on user preference
                            if not filters['show_all_recommendations']:
                                if "NO" in recommendation['verdict'] or "WAIT" in recommendation['verdict']:
                                    continue
                            
                            # Extract Greeks
                            delta = call.get('delta', 0)
                            theta = call.get('theta', 0)
                            gamma = call.get('gamma', 0)
                            vega = call.get('vega', 0)
                            rho = call.get('rho', 0)
                            
                            opportunities.append({
                                'position_key': f"{position['symbol']}_{position['account_type']}",
                                'symbol': position['symbol'],
                                'current_price': current_price,
                                'strike': call['strike'],
                                'strike_desc': strike_desc,
                                'expiration': best_date,
                                'premium': premium,
                                'bid': call['bid'],
                                'ask': call['ask'],
                                'iv': iv,
                                'iv_rank': iv_rank,
                                'growth_score': growth_score,
                                'monthly_yield': monthly_yield,
                                'annual_yield': annual_yield,
                                'shares': position['shares'],
                                'max_contracts': position['shares'] // 100,
                                'growth_category': position['growth_category'],
                                'recommendation': recommendation,
                                'days_to_expiry': (datetime.strptime(best_date, '%Y-%m-%d') - datetime.now()).days,
                                'earnings_date': earnings_date,
                                'days_to_earnings': days_to_earnings,
                                'greeks': {
                                    'delta': delta,
                                    'theta': theta,
                                    'gamma': gamma,
                                    'vega': vega,
                                    'rho': rho
                                }
                            })
                        
        except Exception as e:
            st.error(f"Error scanning {position['symbol']}: {str(e)}")
    
    return sorted(opportunities, key=lambda x: x['monthly_yield'], reverse=True)

def display_opportunity_card(opp: Dict):
    """Display opportunity card using pure Streamlit components with simple background styling."""
    # Format expiration date
    exp_date = datetime.strptime(opp['expiration'], '%Y-%m-%d')
    exp_month = exp_date.strftime('%b').upper()
    exp_day = exp_date.strftime('%d')
    
    # Greeks
    greeks = opp.get('greeks', {})
    delta = abs(greeks.get('delta', 0))
    theta = abs(greeks.get('theta', 0))
    gamma = greeks.get('gamma', 0)
    score = calculate_greeks_score(greeks)

    # Simple container with background
    with st.container():
        # Card header
        st.markdown(f"### {opp['symbol']} - ${opp['strike']:.2f}")
        
        # Expiration and earnings info
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{exp_month} {exp_day}**")
        with col2:
            if opp.get('earnings_date') and opp.get('days_to_earnings', 999) <= 30:
                st.markdown(f"⚠️ **Earnings {opp['earnings_date']} ({opp['days_to_earnings']}d)**")
        
        # Strike description
        st.markdown(f"*{opp['strike_desc']}*")
        
        # Key metrics in 2x3 grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current", f"${opp['current_price']:.2f}")
            st.metric("Premium", f"${opp['premium']:.2f}")
        with col2:
            st.metric("Days", opp['days_to_expiry'])
            st.metric("Monthly", f"{opp['monthly_yield']:.1f}%")
        with col3:
            st.metric("IV Rank", f"{opp['iv_rank']:.0f}%")
            st.metric("Score", f"{score}/10")
        
        # Greeks in 3 columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Δ {delta:.2f}**")
            st.markdown("*ITM*")
        with col2:
            st.markdown(f"**Θ ${theta*100:.1f}**")
            st.markdown("*Daily*")
        with col3:
            st.markdown(f"**Γ {gamma:.3f}**")
            st.markdown("*Risk*")
        
        # Recommendation
        st.markdown("---")
        st.markdown(f"**{opp['recommendation']['verdict']}**")
        st.markdown(f"*{opp['recommendation']['reasoning'][:100]}{'...' if len(opp['recommendation']['reasoning']) > 100 else ''}*")
        
        # Add spacing between cards
        st.markdown("")
        st.markdown("---")
        st.markdown("")

def display_opportunities_section():
    """Display opportunities in a condensed 2-3 column layout using native Streamlit"""
    if hasattr(st.session_state, 'opportunities'):
        if st.session_state.opportunities:
            # Show summary
            st.markdown("### 📊 QUALITY COVERED CALL OPPORTUNITIES")
            st.markdown("*Balanced filtering: Showing good opportunities with clear recommendations*")
            
            # Count recommendations by type
            strong_yes = len([o for o in st.session_state.opportunities if "STRONG YES" in o['recommendation']['verdict']])
            yes = len([o for o in st.session_state.opportunities if "YES" in o['recommendation']['verdict'] and "STRONG" not in o['recommendation']['verdict']])
            maybe = len([o for o in st.session_state.opportunities if "MAYBE" in o['recommendation']['verdict']])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Total Quality Plays", len(st.session_state.opportunities))
            with col2:
                st.metric("🟢 Strong Yes", strong_yes)
            with col3:
                st.metric("🟡 Yes/Maybe", yes + maybe)
            with col4:
                if st.session_state.opportunities:
                    avg_yield = sum(o['monthly_yield'] for o in st.session_state.opportunities) / len(st.session_state.opportunities)
                    st.metric("Avg Monthly Yield", f"{avg_yield:.1f}%")
                else:
                    st.metric("Avg Monthly Yield", "0%")
            
            st.markdown("---")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_verdict = st.selectbox("Filter by Verdict", 
                    ["All", "🟢 Strong Yes Only", "🟢🟡 Yes/Maybe", "🔴 No/Wait"])
            with col2:
                sort_by = st.selectbox("Sort by", 
                    ["Monthly Yield", "IV Rank", "Days to Expiry", "Strike Distance"])
            with col3:
                min_yield = st.slider("Min Monthly Yield %", 0.0, 5.0, 0.0, 0.1)
            
            # Filter and sort opportunities
            filtered_opps = st.session_state.opportunities
            
            if filter_verdict != "All":
                if filter_verdict == "🟢 Strong Yes Only":
                    filtered_opps = [o for o in filtered_opps if "STRONG YES" in o['recommendation']['verdict']]
                elif filter_verdict == "🟢🟡 Yes/Maybe":
                    filtered_opps = [o for o in filtered_opps if "NO" not in o['recommendation']['verdict']]
                elif filter_verdict == "🔴 No/Wait":
                    filtered_opps = [o for o in filtered_opps if "NO" in o['recommendation']['verdict'] or "WAIT" in o['recommendation']['verdict']]
            
            filtered_opps = [o for o in filtered_opps if o['monthly_yield'] >= min_yield]
            
            # Sort
            if sort_by == "Monthly Yield":
                filtered_opps = sorted(filtered_opps, key=lambda x: x['monthly_yield'], reverse=True)
            elif sort_by == "IV Rank":
                filtered_opps = sorted(filtered_opps, key=lambda x: x['iv_rank'], reverse=True)
            elif sort_by == "Days to Expiry":
                filtered_opps = sorted(filtered_opps, key=lambda x: x['days_to_expiry'])
            elif sort_by == "Strike Distance":
                filtered_opps = sorted(filtered_opps, key=lambda x: (x['strike'] - x['current_price']) / x['current_price'])
            
            st.markdown("---")
            
            # Display filtered opportunities in 2-3 column layout
            st.markdown("### 🎯 Covered Call Opportunities")
            
            # Create flexible grid layout
            opportunities_per_row = 3 if len(filtered_opps) > 6 else 2
            
            for i in range(0, len(filtered_opps), opportunities_per_row):
                cols = st.columns(opportunities_per_row)
                
                for j in range(opportunities_per_row):
                    if i + j < len(filtered_opps):
                        with cols[j]:
                            opp = filtered_opps[i + j]
                            opp['_index'] = i + j  # Add index for unique keys
                            # Use native Streamlit components
                            display_opportunity_card(opp)
                
                # Add spacing between rows
                if i + opportunities_per_row < len(filtered_opps):
                    st.markdown("---")
        else:
            st.info("📊 No quality opportunities found with current filters")
            st.markdown("""
            **Our balanced filters require:**
            - ✅ Minimum monthly yield: 0.5% (growth), 0.8% (moderate), 1% (conservative)
            - ✅ IV Rank > 20% (some volatility)
            - ✅ Reasonable bid-ask spreads (<75%)
            - ✅ Meaningful premium (>0.3% of stock price)
            - ✅ 21-35 days to expiration
            - ✅ Avoiding earnings within 14 days
            
            **Possible reasons:**
            - Market volatility is very low
            - Your positions may not have liquid options
            - Options premiums are compressed
            
            💡 **Tips:** Check after market moves, add high-volume stocks, or wait for earnings season
            """)
    else:
        st.info("Click 'Scan for Opportunities' to find covered call trades")

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

def inject_custom_css():
    st.markdown("""
    <style>
    /* SIMPLE, CLEAN CSS - NO COMPLEXITY */
    
    /* Force ALL text to be white */
    * {
        color: #FFFFFF !important;
    }
    
    /* Background */
    .stApp {
        background: #000000;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* All text elements */
    p, div, span, label {
        color: #FFFFFF !important;
    }
    
    /* Metric containers - FORCE WHITE TEXT */
    [data-testid="metric-container"] {
        background: #1A1A1A !important;
        border: 2px solid #444444 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    /* Metric values - LARGE WHITE TEXT */
    [data-testid="metric-container"] > div > div:nth-child(2) {
        color: #FFFFFF !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    /* Metric labels */
    [data-testid="metric-container"] label {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1A1A1A !important;
        border: 2px solid #00D2FF !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
    [data-baseweb="tab"] {
        color: #FFFFFF !important;
        background: #1A1A1A !important;
    }
    
    [aria-selected="true"] {
        background: #00D2FF !important;
        color: #000000 !important;
    }
    
    /* Input fields */
    input, select, textarea {
        background: #1A1A1A !important;
        border: 2px solid #444444 !important;
        color: #FFFFFF !important;
    }
    
    /* Alerts */
    .stAlert {
        background: #1A1A1A !important;
        border: 2px solid #444444 !important;
        color: #FFFFFF !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Main container */
    .block-container {
        background: #000000 !important;
        padding: 20px !important;
    }
    
    /* Force ALL markdown text white */
    .stMarkdown, .stMarkdown * {
        color: #FFFFFF !important;
    }
    
    /* Override any inline styles */
    [style*="color"] {
        color: #FFFFFF !important;
    }
    
    /* Force all containers to have white text */
    .element-container * {
        color: #FFFFFF !important;
    }
    
    /* Nuclear option - catch everything */
    *, *::before, *::after {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Fix for select, input, and slider controls - MORE AGGRESSIVE */
    .stSelectbox, .stTextInput, .stSlider, .stNumberInput, .stRadio, .stDateInput, .stMultiSelect, .stTextArea {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    .stSelectbox input, .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 2px solid #444444 !important;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00D2FF, #FF6B35) !important;
    }
    .stSlider > div > div {
        background: #1A1A1A !important;
    }
    
    /* MORE AGGRESSIVE FILTER FIXES */
    .stSelectbox > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    .stSelectbox > div > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    .stSelectbox > div > div > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    .stSelectbox > div > div > div > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    
    /* Slider specific fixes */
    .stSlider > div > div > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    .stSlider > div > div > div > div > div {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    
    /* Force all form elements to have dark background */
    [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="textarea"] {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    
    /* Override any white backgrounds in form elements */
    [style*="background-color: white"], [style*="background: white"] {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Inject custom CSS
    inject_custom_css()
    
    # Simple header
    st.title("⚡ QUANTUM INCOME COMMAND CENTER ⚡")
    
    # Mission status
    st.markdown("**MISSION: GENERATE $2-5K MONTHLY INCOME | TARGET: $60K MARGIN DEBT ELIMINATION | STATUS: ACTIVE**")
    
    initialize_session_state()
    
    alerts = check_21_50_7_alerts()
    if alerts:
        display_alerts(alerts)
    
    display_monthly_progress()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Opportunities", "📊 Positions", "🚀 Growth Scanner", "👁️ WatchList"])
    
    with tab1:
        # Show positions that can't sell calls
        insufficient_shares = [p for p in st.session_state.positions if p['shares'] < 100]
        if insufficient_shares:
            with st.expander(f"ℹ️ {len(insufficient_shares)} positions with <100 shares (can't sell calls)", expanded=False):
                for pos in insufficient_shares:
                    st.write(f"• **{pos['symbol']}**: {pos['shares']} shares (need {100 - pos['shares']} more)")
        
        # Add filter controls BEFORE scanning
        with st.expander("⚙️ Adjust Scan Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Minimum Monthly Yield %**")
                min_yield_growth = st.slider("Growth Stocks", 0.0, 3.0, 0.5, 0.1, key="yield_growth")
                min_yield_moderate = st.slider("Moderate Stocks", 0.0, 3.0, 0.8, 0.1, key="yield_moderate")
                min_yield_conservative = st.slider("Conservative Stocks", 0.0, 3.0, 1.0, 0.1, key="yield_conservative")
            
            with col2:
                st.markdown("**Volatility & Spreads**")
                min_iv_rank = st.slider("Min IV Rank %", 0, 50, 20, 5)
                max_spread = st.slider("Max Bid-Ask Spread %", 25, 150, 75, 25)
                min_premium_pct = st.slider("Min Premium (% of stock)", 0.1, 1.0, 0.3, 0.1)
            
            with col3:
                st.markdown("**Other Filters**")
                include_earnings = st.checkbox("Include stocks with earnings <14 days", value=False)
                show_all_recommendations = st.checkbox("Show NO/WAIT recommendations", value=True)
                days_range = st.select_slider("Days to Expiration Range", 
                    options=[(14,21), (21,35), (21,45), (30,60)],
                    value=(21,35),
                    format_func=lambda x: f"{x[0]}-{x[1]} days")
            
            # Store filter settings
            st.session_state.scan_filters = {
                'min_yield_growth': min_yield_growth,
                'min_yield_moderate': min_yield_moderate,
                'min_yield_conservative': min_yield_conservative,
                'min_iv_rank': min_iv_rank,
                'max_spread': max_spread / 100,  # Convert to decimal
                'min_premium_pct': min_premium_pct / 100,  # Convert to decimal
                'include_earnings': include_earnings,
                'show_all_recommendations': show_all_recommendations,
                'days_range': days_range
            }
        
        if st.button("🔄 Scan for Opportunities"):
            with st.spinner("Scanning positions..."):
                opportunities = scan_covered_call_opportunities()
                st.session_state.opportunities = opportunities
        
        # Use the new display function
        display_opportunities_section()
        
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
        # Options tracking section
        with st.expander("📊 Active Options Positions", expanded=True):
            display_active_options()
        
        # Toggle between single add and bulk import
        import_mode = st.radio("Add Method", ["➕ Single Position", "📋 Bulk Import", "🎯 Add Option Trade"], horizontal=True)
        
        if import_mode == "➕ Single Position":
            add_position()
        elif import_mode == "📋 Bulk Import":
            bulk_import_positions()
        else:
            add_option_trade()
            
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
                
                # Display column headers
                header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7, header_col8 = st.columns([1.5, 1, 1, 1, 1, 1, 3, 1])
                with header_col1:
                    st.markdown("**Symbol**")
                with header_col2:
                    st.markdown("**Added**")
                with header_col3:
                    st.markdown("**Entry**")
                with header_col4:
                    st.markdown("**Current**")
                with header_col5:
                    st.markdown("**Change**")
                with header_col6:
                    st.markdown("**Score**")
                with header_col7:
                    st.markdown("**Notes**")
                with header_col8:
                    st.markdown("**Action**")
                
                st.markdown("---")
                
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