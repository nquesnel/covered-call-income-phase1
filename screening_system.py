import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import concurrent.futures
import time
try:
    from tradingview_screener import Query, Column as col
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False
    print("TradingView screener not available, using fallback method")

class GrowthScreeningSystem:
    """Two-tier growth stock screening system with self-improvement capabilities"""
    
    def __init__(self):
        self.data_dir = "data"
        self.screening_history_file = os.path.join(self.data_dir, "screening_history.json")
        self.screening_results_file = os.path.join(self.data_dir, "screening_results.json")
        self.refinement_log_file = os.path.join(self.data_dir, "refinement_log.json")
        
        # Initialize data storage
        os.makedirs(self.data_dir, exist_ok=True)
        self.screening_history = self.load_json(self.screening_history_file)
        self.refinement_log = self.load_json(self.refinement_log_file)
        
        # Tier 1: High Conviction Criteria (adjustable based on results)
        self.tier1_criteria = {
            "min_revenue_growth": 0.40,  # 40% YoY
            "min_revenue_acceleration": 0.05,  # 5% QoQ acceleration
            "max_peg_ratio": 1.0,
            "min_gross_margin": 0.60,  # 60%
            "max_price_from_high": 0.25,  # Within 25% of 52-week high
            "min_institutional_momentum": 0.05,  # 5% increase in ownership
            "min_cash_runway_months": 12  # If unprofitable
        }
        
        # Tier 2: Expanded Screen Criteria
        self.tier2_criteria = {
            "min_revenue_growth": 0.25,  # 25% YoY
            "min_eps_growth": 0.20,  # 20% YoY
            "min_rule_of_40": 40,  # Growth + profit margin
            "min_relative_strength": 1.20,  # vs S&P 500
            "min_analyst_revision_trend": 0  # Positive revisions
        }
        
        # High-growth stock indicators (learned from research)
        self.growth_patterns = {
            "platform_effect": ["AMZN", "GOOGL", "META", "NVDA"],
            "secular_trend": ["AI", "Cloud", "EV", "Renewable", "Biotech"],
            "founder_led": ["TSLA", "NVDA", "META"],
            "high_nrr": 1.20,  # 120% net revenue retention for SaaS
            "tam_expansion": ["new_products", "new_markets", "acquisitions"]
        }
        
        # Early Innings Criteria - Find next NVDA at $10B, not $4T
        self.early_innings_criteria = {
            "max_market_cap": 50_000_000_000,  # <$50B
            "min_market_cap": 1_000_000_000,   # >$1B (avoid penny stocks)
            "min_revenue_growth": 0.30,         # 30%+ growth minimum
            "min_gross_margin": 0.50,           # High margin business
            "max_ps_ratio": 20,                 # Not crazy expensive
            "min_insider_ownership": 0.10,      # 10%+ insider ownership
            "min_revenue": 100_000_000,         # $100M+ revenue (real business)
            "max_age_years": 15                 # IPO'd in last 15 years
        }
        
    def load_json(self, filepath: str) -> Dict:
        """Load JSON data from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def save_json(self, filepath: str, data: Dict):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def calculate_revenue_growth(self, ticker: yf.Ticker) -> Tuple[float, float]:
        """Calculate YoY and QoQ revenue growth"""
        try:
            # Get quarterly financials
            financials = ticker.quarterly_financials
            if financials.empty:
                return 0, 0
            
            # Get revenue (Total Revenue row)
            revenue_row = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            if revenue_row is None:
                return 0, 0
            
            # YoY growth (compare to same quarter last year)
            if len(revenue_row) >= 5:
                current = revenue_row.iloc[0]
                year_ago = revenue_row.iloc[4]
                yoy_growth = (current - year_ago) / year_ago if year_ago > 0 else 0
            else:
                yoy_growth = 0
            
            # QoQ growth
            if len(revenue_row) >= 2:
                current = revenue_row.iloc[0]
                last_quarter = revenue_row.iloc[1]
                qoq_growth = (current - last_quarter) / last_quarter if last_quarter > 0 else 0
            else:
                qoq_growth = 0
            
            return yoy_growth, qoq_growth
        except:
            return 0, 0
    
    def calculate_peg_ratio(self, info: Dict) -> float:
        """Calculate PEG ratio (PE / Growth Rate)"""
        try:
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
            peg_ratio = info.get('pegRatio', 0)
            
            if peg_ratio > 0:
                return peg_ratio
            elif pe_ratio > 0:
                # Estimate growth from analyst estimates
                growth_estimate = info.get('earningsQuarterlyGrowth', 0)
                if growth_estimate > 0:
                    return pe_ratio / (growth_estimate * 100)
            return 999  # High number if can't calculate
        except:
            return 999
    
    def check_institutional_momentum(self, ticker: yf.Ticker) -> float:
        """Check if institutional ownership is increasing"""
        try:
            # Get institutional holders
            inst_holders = ticker.institutional_holders
            if inst_holders.empty:
                return 0
            
            # Look for recent changes (this is simplified - in reality we'd track over time)
            # For now, return a placeholder
            return 0.05  # Would need to track historical data
        except:
            return 0
    
    def calculate_relative_strength(self, symbol: str, period: str = "6mo") -> float:
        """Calculate stock performance vs S&P 500"""
        try:
            stock = yf.Ticker(symbol)
            spy = yf.Ticker("SPY")
            
            # Get historical data
            stock_hist = stock.history(period=period)
            spy_hist = spy.history(period=period)
            
            if stock_hist.empty or spy_hist.empty:
                return 0
            
            # Calculate returns
            stock_return = (stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0]) - 1
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1
            
            # Relative strength
            relative_strength = (1 + stock_return) / (1 + spy_return)
            return relative_strength
        except:
            return 0
    
    def calculate_analyst_upside(self, info: Dict) -> float:
        """Calculate analyst price target upside"""
        try:
            current_price = info.get('currentPrice', 0)
            target_price = info.get('targetMeanPrice', 0)
            
            if current_price > 0 and target_price > 0:
                return (target_price - current_price) / current_price
            return 0
        except:
            return 0
    
    def calculate_growth_probability_score(self, symbol: str, screening_data: Dict) -> Dict:
        """Calculate probability of being a high-growth winner (0-100)"""
        score = 0
        factors = []
        
        # Get market cap for scale-adjusted scoring
        market_cap = screening_data.get('market_cap', 0)
        
        # 1. Scale-Adjusted Revenue Growth (0-25 points)
        revenue_growth = screening_data.get('revenue_growth_yoy', 0)
        
        # Different standards by market cap
        if market_cap > 1_000_000_000_000:  # >$1T mega cap
            if revenue_growth > 0.15:  # >15% is exceptional at this scale
                score += 25
                factors.append(f"Exceptional growth for mega-cap ({revenue_growth:.0%})")
            elif revenue_growth > 0.10:
                score += 20
                factors.append(f"Strong growth for mega-cap ({revenue_growth:.0%})")
            elif revenue_growth > 0.05:
                score += 10
                factors.append(f"Solid growth for mega-cap ({revenue_growth:.0%})")
                
        elif market_cap > 100_000_000_000:  # >$100B large cap
            if revenue_growth > 0.25:  # >25%
                score += 25
                factors.append(f"Exceptional growth for large-cap ({revenue_growth:.0%})")
            elif revenue_growth > 0.20:
                score += 20
                factors.append(f"Strong growth for large-cap ({revenue_growth:.0%})")
            elif revenue_growth > 0.15:
                score += 15
                factors.append(f"Good growth for large-cap ({revenue_growth:.0%})")
            elif revenue_growth > 0.10:
                score += 10
                factors.append(f"Decent growth ({revenue_growth:.0%})")
                
        else:  # <$100B - need higher growth
            if revenue_growth > 0.50:  # >50%
                score += 25
                factors.append(f"Hyper growth ({revenue_growth:.0%})")
            elif revenue_growth > 0.30:
                score += 20
                factors.append(f"Very high growth ({revenue_growth:.0%})")
            elif revenue_growth > 0.20:
                score += 15
                factors.append(f"High growth ({revenue_growth:.0%})")
            else:
                score += max(0, revenue_growth * 30)  # Scale it
        
        # 2. AI/Technology Leadership Bonus (0-20 points)
        sector = screening_data.get('sector', '')
        ai_leaders = ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'PLTR']
        cloud_leaders = ['AMZN', 'MSFT', 'GOOGL', 'SNOW', 'DDOG', 'NET', 'CRWD']
        
        if symbol in ai_leaders:
            score += 20
            factors.append("AI/Tech leadership position")
        elif symbol in cloud_leaders:
            score += 15
            factors.append("Cloud/SaaS leadership")
        elif sector in ['Technology', 'Communication Services']:
            score += 10
            factors.append("High-growth sector")
        
        # 3. Profitability & Quality (0-20 points)
        gross_margin = screening_data.get('gross_margin', 0)
        operating_margin = screening_data.get('operating_margin', 0)
        
        # For mega caps, profitability is key
        if market_cap > 500_000_000_000:  # >$500B
            if gross_margin > 0.50 and operating_margin > 0.25:
                score += 20
                factors.append("Excellent profitability metrics")
            elif gross_margin > 0.40 and operating_margin > 0.20:
                score += 15
                factors.append("Strong profitability")
            elif gross_margin > 0.30 and operating_margin > 0.15:
                score += 10
                factors.append("Good profitability")
        else:
            # Smaller companies get points for high gross margins
            if gross_margin > 0.70:
                score += 15
                factors.append(f"Exceptional gross margins ({gross_margin:.0%})")
            elif gross_margin > 0.60:
                score += 10
                factors.append(f"Strong gross margins ({gross_margin:.0%})")
            elif gross_margin > 0.50:
                score += 5
                factors.append(f"Good gross margins ({gross_margin:.0%})")
        
        # 4. Momentum & Technical Strength (0-15 points)
        relative_strength = screening_data.get('relative_strength', 0)
        if relative_strength > 1.30:  # 30%+ outperformance
            score += 15
            factors.append(f"Strong momentum ({relative_strength:.2f}x vs SPY)")
        elif relative_strength > 1.15:
            score += 10
            factors.append(f"Good momentum ({relative_strength:.2f}x vs SPY)")
        elif relative_strength > 1.05:
            score += 5
            factors.append(f"Outperforming market ({relative_strength:.2f}x)")
        
        # 5. Future Growth Potential (0-20 points)
        # Check analyst estimates, TAM expansion, etc.
        target_upside = screening_data.get('analyst_upside', 0)
        peg_ratio = screening_data.get('peg_ratio', 999)
        
        if target_upside > 0.30:  # >30% upside
            score += 10
            factors.append(f"Strong analyst conviction ({target_upside:.0%} upside)")
        elif target_upside > 0.15:
            score += 5
            factors.append(f"Positive analyst outlook ({target_upside:.0%} upside)")
        
        if peg_ratio < 1.0 and peg_ratio > 0:
            score += 10
            factors.append(f"Attractive growth valuation (PEG {peg_ratio:.2f})")
        elif peg_ratio < 1.5:
            score += 5
            factors.append(f"Reasonable valuation (PEG {peg_ratio:.2f})")
        
        # 6. Special Situations & Catalysts
        # Acceleration is huge signal
        if screening_data.get('revenue_acceleration', False):
            score += 10
            factors.append("Revenue growth accelerating!")
        
        # New product cycles, market expansion
        catalyst_stocks = ['NVDA', 'TSLA', 'META', 'GOOGL', 'MSFT', 'AAPL', 'AMD']
        if symbol in catalyst_stocks:
            score += 5
            factors.append("Major growth catalysts present")
        
        # Cap score at 100
        score = min(100, max(0, score))
        
        # Determine growth profile
        if score >= 80:
            profile = "EXTREME GROWTH"
            action = "Strong buy, NO covered calls"
        elif score >= 70:
            profile = "HIGH GROWTH"
            action = "Accumulate, very limited CCs only"
        elif score >= 60:
            profile = "SOLID GROWTH"
            action = "Good position, conservative CCs OK"
        elif score >= 40:
            profile = "MODERATE GROWTH"
            action = "Standard CC strategies appropriate"
        else:
            profile = "VALUE/INCOME"
            action = "Aggressive CC income generation"
        
        return {
            "score": score,
            "profile": profile,
            "action": action,
            "confidence": "HIGH" if score >= 70 else "MEDIUM" if score >= 50 else "SPECULATION",
            "factors": factors
        }
    
    def screen_tier1(self, symbol: str) -> Optional[Dict]:
        """Run Tier 1 High Conviction screen on a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Calculate metrics
            yoy_growth, qoq_growth = self.calculate_revenue_growth(ticker)
            peg_ratio = self.calculate_peg_ratio(info)
            
            # Check Tier 1 criteria
            checks = {
                "revenue_growth": yoy_growth >= self.tier1_criteria["min_revenue_growth"],
                "revenue_acceleration": qoq_growth > 0.05,
                "peg_ratio": peg_ratio <= self.tier1_criteria["max_peg_ratio"],
                "gross_margin": info.get('grossMargins', 0) >= self.tier1_criteria["min_gross_margin"],
                "near_high": self.check_near_52w_high(ticker, self.tier1_criteria["max_price_from_high"]),
                "institutional_momentum": True,  # Simplified for now
                "cash_runway": self.check_cash_runway(info) >= self.tier1_criteria["min_cash_runway_months"]
            }
            
            # Must meet ALL criteria for Tier 1
            if all(checks.values()):
                screening_data = {
                    "symbol": symbol,
                    "tier": 1,
                    "revenue_growth_yoy": yoy_growth,
                    "revenue_growth_qoq": qoq_growth,
                    "revenue_acceleration": qoq_growth > 0.05,
                    "peg_ratio": peg_ratio,
                    "gross_margin": info.get('grossMargins', 0),
                    "operating_margin": info.get('operatingMargins', 0),
                    "market_cap": info.get('marketCap', 0),
                    "sector": info.get('sector', 'Unknown'),
                    "relative_strength": self.calculate_relative_strength(symbol),
                    "analyst_upside": self.calculate_analyst_upside(info),
                    "criteria_met": [k for k, v in checks.items() if v],
                    "screened_date": datetime.now().isoformat()
                }
                
                # Calculate growth probability
                probability = self.calculate_growth_probability_score(symbol, screening_data)
                screening_data.update(probability)
                
                return screening_data
                
        except Exception as e:
            print(f"Error screening {symbol}: {e}")
        
        return None
    
    def screen_tier2(self, symbol: str) -> Optional[Dict]:
        """Run Tier 2 Expanded screen on a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Calculate metrics
            yoy_growth, _ = self.calculate_revenue_growth(ticker)
            relative_strength = self.calculate_relative_strength(symbol)
            
            # Rule of 40 calculation (for SaaS companies)
            growth_rate = yoy_growth * 100
            profit_margin = info.get('profitMargins', 0) * 100
            rule_of_40 = growth_rate + profit_margin
            
            # Check Tier 2 criteria (need 5+ of these)
            checks = {
                "revenue_growth": yoy_growth >= self.tier2_criteria["min_revenue_growth"],
                "eps_growth": self.check_eps_growth(ticker) >= self.tier2_criteria["min_eps_growth"],
                "rule_of_40": rule_of_40 >= self.tier2_criteria["min_rule_of_40"],
                "relative_strength": relative_strength >= self.tier2_criteria["min_relative_strength"],
                "analyst_revisions": True,  # Simplified for now
                "insider_buying": True,  # Simplified for now
                "volume_pattern": True  # Simplified for now
            }
            
            # Need at least 5 criteria met for Tier 2
            if sum(checks.values()) >= 5:
                screening_data = {
                    "symbol": symbol,
                    "tier": 2,
                    "revenue_growth_yoy": yoy_growth,
                    "relative_strength": relative_strength,
                    "rule_of_40": rule_of_40,
                    "gross_margin": info.get('grossMargins', 0),
                    "operating_margin": info.get('operatingMargins', 0),
                    "market_cap": info.get('marketCap', 0),
                    "sector": info.get('sector', 'Unknown'),
                    "analyst_upside": self.calculate_analyst_upside(info),
                    "peg_ratio": self.calculate_peg_ratio(info),
                    "criteria_met": [k for k, v in checks.items() if v],
                    "screened_date": datetime.now().isoformat()
                }
                
                # Calculate growth probability
                probability = self.calculate_growth_probability_score(symbol, screening_data)
                screening_data.update(probability)
                
                return screening_data
                
        except Exception as e:
            print(f"Error screening {symbol}: {e}")
        
        return None
    
    def check_near_52w_high(self, ticker: yf.Ticker, max_distance: float) -> bool:
        """Check if stock is within X% of 52-week high"""
        try:
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
            
            if current_price and fifty_two_week_high:
                distance = (fifty_two_week_high - current_price) / fifty_two_week_high
                return distance <= max_distance
        except:
            pass
        return False
    
    def check_cash_runway(self, info: Dict) -> float:
        """Calculate months of cash runway for unprofitable companies"""
        try:
            if info.get('profitMargins', 0) > 0:
                return 999  # Profitable companies don't need runway
            
            cash = info.get('totalCash', 0)
            burn_rate = abs(info.get('freeCashflow', 0)) / 4  # Quarterly to monthly
            
            if burn_rate > 0:
                return cash / burn_rate
        except:
            pass
        return 0
    
    def check_eps_growth(self, ticker: yf.Ticker) -> float:
        """Check EPS growth rate"""
        try:
            info = ticker.info
            return info.get('earningsQuarterlyGrowth', 0)
        except:
            return 0
    
    def screen_universe(self, symbols: List[str], tier: int = 1) -> List[Dict]:
        """Screen a list of symbols"""
        results = []
        
        for symbol in symbols:
            print(f"Screening {symbol}...")
            
            if tier == 1:
                result = self.screen_tier1(symbol)
            else:
                result = self.screen_tier2(symbol)
            
            if result:
                results.append(result)
                # Log the screening event
                self.log_screening_event(result)
        
        # Sort by growth probability score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Save results
        self.save_json(self.screening_results_file, {
            "screen_date": datetime.now().isoformat(),
            "tier": tier,
            "results": results
        })
        
        return results
    
    def log_screening_event(self, screening_data: Dict):
        """Log screening event for future analysis"""
        event_id = f"{screening_data['symbol']}_{datetime.now().strftime('%Y%m%d')}"
        
        self.screening_history[event_id] = {
            **screening_data,
            "outcome_30d": None,  # To be filled later
            "outcome_90d": None,
            "outcome_180d": None,
            "success": None  # To be determined
        }
        
        self.save_json(self.screening_history_file, self.screening_history)
    
    def update_outcomes(self):
        """Update outcomes for historical screens (run periodically)"""
        for event_id, event in self.screening_history.items():
            if event['outcome_180d'] is not None:
                continue  # Already fully updated
            
            symbol = event['symbol']
            screen_date = datetime.fromisoformat(event['screened_date'])
            days_since = (datetime.now() - screen_date).days
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=screen_date, end=datetime.now())
                
                if not hist.empty:
                    start_price = hist['Close'].iloc[0]
                    current_price = hist['Close'].iloc[-1]
                    
                    # Update based on time elapsed
                    if days_since >= 30 and event['outcome_30d'] is None:
                        price_30d = hist['Close'].iloc[min(30, len(hist)-1)]
                        event['outcome_30d'] = (price_30d - start_price) / start_price
                    
                    if days_since >= 90 and event['outcome_90d'] is None:
                        price_90d = hist['Close'].iloc[min(90, len(hist)-1)]
                        event['outcome_90d'] = (price_90d - start_price) / start_price
                    
                    if days_since >= 180 and event['outcome_180d'] is None:
                        price_180d = hist['Close'].iloc[min(180, len(hist)-1)]
                        event['outcome_180d'] = (price_180d - start_price) / start_price
                        
                        # Determine success (>50% gain in 180 days for Tier 1, >25% for Tier 2)
                        if event['tier'] == 1:
                            event['success'] = event['outcome_180d'] > 0.50
                        else:
                            event['success'] = event['outcome_180d'] > 0.25
            except:
                pass
        
        self.save_json(self.screening_history_file, self.screening_history)
    
    def analyze_and_refine(self) -> Dict:
        """Analyze screening history and suggest refinements"""
        # Update outcomes first
        self.update_outcomes()
        
        # Analyze what's working
        analysis = {
            "total_screens": len(self.screening_history),
            "success_rate": 0,
            "best_criteria": {},
            "worst_criteria": {},
            "recommended_adjustments": []
        }
        
        # Calculate success rates
        completed = [e for e in self.screening_history.values() if e['success'] is not None]
        if completed:
            successes = [e for e in completed if e['success']]
            analysis['success_rate'] = len(successes) / len(completed)
            
            # Analyze which criteria correlate with success
            criteria_success = {}
            for event in completed:
                for criteria in event.get('criteria_met', []):
                    if criteria not in criteria_success:
                        criteria_success[criteria] = {"success": 0, "total": 0}
                    criteria_success[criteria]["total"] += 1
                    if event['success']:
                        criteria_success[criteria]["success"] += 1
            
            # Calculate success rate by criteria
            for criteria, stats in criteria_success.items():
                if stats['total'] > 5:  # Need meaningful sample size
                    success_rate = stats['success'] / stats['total']
                    if success_rate > 0.7:
                        analysis['best_criteria'][criteria] = success_rate
                    elif success_rate < 0.3:
                        analysis['worst_criteria'][criteria] = success_rate
            
            # Generate recommendations
            if analysis['success_rate'] < 0.5:
                analysis['recommended_adjustments'].append(
                    "Consider loosening revenue growth requirement to 35%"
                )
            
            if 'peg_ratio' in analysis['worst_criteria']:
                analysis['recommended_adjustments'].append(
                    "PEG ratio may be too restrictive - consider raising to 1.5"
                )
        
        # Log the analysis
        self.refinement_log[datetime.now().isoformat()] = analysis
        self.save_json(self.refinement_log_file, self.refinement_log)
        
        return analysis
    
    def screen_early_innings(self, symbol: str) -> Optional[Dict]:
        """Screen for early-stage growth stocks (1st-3rd inning)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Basic filters
            market_cap = info.get('marketCap', 0)
            if market_cap > self.early_innings_criteria["max_market_cap"]:
                return None  # Too big
            if market_cap < self.early_innings_criteria["min_market_cap"]:
                return None  # Too small
            
            # Revenue and growth checks
            revenue = info.get('totalRevenue', 0)
            if revenue < self.early_innings_criteria["min_revenue"]:
                return None  # Not enough revenue yet
            
            yoy_growth, qoq_growth = self.calculate_revenue_growth(ticker)
            if yoy_growth < self.early_innings_criteria["min_revenue_growth"]:
                return None  # Not growing fast enough
            
            # Quality checks
            gross_margin = info.get('grossMargins', 0)
            if gross_margin < self.early_innings_criteria["min_gross_margin"]:
                return None  # Low margin business
            
            # Valuation check (Price/Sales)
            ps_ratio = market_cap / revenue if revenue > 0 else 999
            if ps_ratio > self.early_innings_criteria["max_ps_ratio"]:
                return None  # Too expensive relative to sales
            
            # Calculate early innings score (different from growth score)
            early_score = self.calculate_early_innings_score(symbol, info, yoy_growth, qoq_growth)
            
            # Calculate conviction score
            conviction = self.calculate_conviction_score(
                early_score['early_innings_score'], 
                early_score['factors'],
                info
            )
            
            # Get company description
            company_description = self.get_company_description(info)
            
            # If we made it here, it's a candidate
            screening_data = {
                "symbol": symbol,
                "tier": "early_innings",
                "market_cap": market_cap,
                "revenue": revenue,
                "revenue_growth_yoy": yoy_growth,
                "revenue_growth_qoq": qoq_growth,
                "gross_margin": gross_margin,
                "ps_ratio": ps_ratio,
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "company_description": company_description,
                "employees": info.get('fullTimeEmployees', 0),
                "founded": info.get('foundedYear', 'Unknown'),
                "website": info.get('website', ''),
                "relative_strength": self.calculate_relative_strength(symbol, period="3mo"),
                "insider_ownership": info.get('heldPercentInsiders', 0),
                "institutional_ownership": info.get('heldPercentInstitutions', 0),
                "screened_date": datetime.now().isoformat(),
                **early_score,
                **conviction
            }
            
            return screening_data
            
        except Exception as e:
            print(f"Error screening {symbol}: {e}")
        
        return None
    
    def get_company_description(self, info: Dict) -> str:
        """Get company description from Yahoo Finance"""
        try:
            return info.get('longBusinessSummary', info.get('description', 'No description available'))
        except:
            return "No description available"
    
    def calculate_conviction_score(self, early_score: int, factors: List[str], info: Dict) -> Dict:
        """Calculate conviction score based on multiple factors"""
        conviction_points = 0
        conviction_factors = []
        
        # Base it on early innings score
        if early_score >= 80:
            conviction_points += 40
            conviction_factors.append("Exceptional growth metrics")
        elif early_score >= 65:
            conviction_points += 30
            conviction_factors.append("Strong growth potential")
        elif early_score >= 50:
            conviction_points += 20
            conviction_factors.append("Solid fundamentals")
        else:
            conviction_points += 10
        
        # Revenue consistency
        if "Accelerating" in str(factors):
            conviction_points += 20
            conviction_factors.append("Accelerating growth trend")
        
        # Management quality (insider ownership proxy)
        if info.get('heldPercentInsiders', 0) > 0.15:
            conviction_points += 15
            conviction_factors.append("Strong insider alignment")
        
        # Market opportunity
        if any(x in str(factors) for x in ["AI", "Cloud", "Cyber", "Bio"]):
            conviction_points += 15
            conviction_factors.append("Massive TAM opportunity")
        
        # Profitability trajectory
        if info.get('grossMargins', 0) > 0.70:
            conviction_points += 10
            conviction_factors.append("Excellent unit economics")
        
        # Cap at 100
        conviction_points = min(100, conviction_points)
        
        # Determine conviction level
        if conviction_points >= 85:
            level = "EXTREME"
            action = "Back up the truck!"
        elif conviction_points >= 70:
            level = "HIGH"
            action = "Build significant position"
        elif conviction_points >= 55:
            level = "MODERATE"
            action = "Start position, add on dips"
        else:
            level = "SPECULATIVE"
            action = "Small starter position only"
        
        return {
            "conviction_score": conviction_points,
            "conviction_level": level,
            "conviction_action": action,
            "conviction_factors": conviction_factors
        }
    
    def calculate_early_innings_score(self, symbol: str, info: Dict, yoy_growth: float, qoq_growth: float) -> Dict:
        """Calculate early innings potential score (0-100)"""
        score = 0
        factors = []
        
        # 1. Hypergrowth (0-30 points) - We want EXPLOSIVE growth
        if yoy_growth > 1.0:  # >100% growth
            score += 30
            factors.append(f"Hypergrowth ({yoy_growth:.0%} YoY)")
        elif yoy_growth > 0.70:  # >70% growth
            score += 25
            factors.append(f"Exceptional growth ({yoy_growth:.0%} YoY)")
        elif yoy_growth > 0.50:  # >50% growth
            score += 20
            factors.append(f"Very high growth ({yoy_growth:.0%} YoY)")
        elif yoy_growth > 0.30:  # >30% growth
            score += 15
            factors.append(f"Strong growth ({yoy_growth:.0%} YoY)")
        
        # 2. Acceleration (0-20 points) - Is growth speeding up?
        if qoq_growth > yoy_growth / 4:  # QoQ exceeding average
            score += 20
            factors.append("Growth is ACCELERATING!")
        elif qoq_growth > 0.10:  # >10% QoQ
            score += 10
            factors.append("Strong quarterly momentum")
        
        # 3. TAM and Market Position (0-20 points)
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # High-growth industries
        if any(trend in industry.lower() for trend in ['artificial intelligence', 'cloud', 'software', 'cyber', 'biotech', 'renewable']):
            score += 15
            factors.append(f"High-growth industry: {industry}")
        elif sector in ['Technology', 'Healthcare']:
            score += 10
            factors.append(f"Growth sector: {sector}")
        
        # Small enough to have runway
        market_cap = info.get('marketCap', 0)
        if market_cap < 10_000_000_000:  # <$10B
            score += 5
            factors.append("Massive growth runway (<$10B cap)")
        
        # 4. Business Quality (0-15 points)
        gross_margin = info.get('grossMargins', 0)
        if gross_margin > 0.80:  # >80% margins
            score += 15
            factors.append(f"Software-like margins ({gross_margin:.0%})")
        elif gross_margin > 0.70:
            score += 10
            factors.append(f"Excellent margins ({gross_margin:.0%})")
        elif gross_margin > 0.60:
            score += 5
            factors.append(f"Strong margins ({gross_margin:.0%})")
        
        # 5. Insider Confidence (0-15 points)
        insider_ownership = info.get('heldPercentInsiders', 0)
        if insider_ownership > 0.20:  # >20% insider owned
            score += 15
            factors.append(f"High insider ownership ({insider_ownership:.0%})")
        elif insider_ownership > 0.10:
            score += 10
            factors.append(f"Good insider ownership ({insider_ownership:.0%})")
        elif insider_ownership > 0.05:
            score += 5
            factors.append(f"Decent insider ownership ({insider_ownership:.0%})")
        
        # Special recognition for potential category creators
        category_creators = ['NET', 'DDOG', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'BILL', 'HUBS', 'VEEV', 'TEAM']
        if symbol in category_creators:
            score += 10
            factors.append("Category-defining company")
        
        # Determine potential
        if score >= 80:
            potential = "🚀 FUTURE GIANT"
            recommendation = "Strong accumulate - this could 10x"
        elif score >= 65:
            potential = "💎 HIDDEN GEM"
            recommendation = "High potential - build position carefully"
        elif score >= 50:
            potential = "🌱 EMERGING GROWTH"
            recommendation = "Promising - watch closely"
        else:
            potential = "👀 EARLY STAGE"
            recommendation = "Interesting but needs more proof"
        
        return {
            "early_innings_score": score,
            "potential": potential,
            "recommendation": recommendation,
            "factors": factors
        }
    
    def get_all_us_stocks(self, min_market_cap: float = 1_000_000_000) -> List[str]:
        """Get all US stocks above minimum market cap using TradingView"""
        try:
            if not TRADINGVIEW_AVAILABLE:
                print("TradingView not available, using S&P 500 fallback")
                return self.get_sp500_symbols()
                
            print(f"Fetching all US stocks above ${min_market_cap/1e9:.1f}B market cap...")
            
            query = (Query()
                    .select('ticker', 'market_cap_basic', 'volume')
                    .where(
                        col('market_cap_basic') > min_market_cap,
                        col('type').in_(['stock']),
                        col('exchange').in_(['NASDAQ', 'NYSE', 'AMEX']),
                        col('subtype').in_(['common', 'foreign-issuer']),
                        col('volume') > 100000,  # Min volume filter
                    )
                    .order_by('market_cap_basic', ascending=False)
                    .limit(2000)  # Get top 2000 stocks
                    )
            
            data = query.get_scanner_data()
            symbols = [row['ticker'] for row in data[1] if row.get('ticker')]
            
            print(f"Found {len(symbols)} stocks above ${min_market_cap/1e9:.1f}B")
            return symbols
            
        except Exception as e:
            print(f"Error fetching all US stocks: {e}")
            # Fall back to S&P 500 if screener fails
            return self.get_sp500_symbols()
    
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols as a starting universe"""
        # In production, you'd get this from a reliable source
        # For now, return a sample of high-growth candidates
        return [
            "NVDA", "TSLA", "META", "GOOGL", "AMZN", "MSFT", "AAPL",
            "AMD", "NFLX", "CRM", "ADBE", "NOW", "SHOP", "SQ", "DDOG",
            "NET", "CRWD", "PANW", "SNOW", "PLTR", "ABNB", "UBER"
        ]
    
    def get_early_innings_candidates(self) -> List[str]:
        """Get ALL stocks in the $1B-$50B market cap range using TradingView screener"""
        try:
            # Try to load cached universe (refreshed daily)
            cache_file = os.path.join(self.data_dir, "market_universe_cache.json")
            
            # Check if cache exists and is less than 24 hours old
            if os.path.exists(cache_file):
                cache_data = self.load_json(cache_file)
                if cache_data and 'timestamp' in cache_data:
                    cache_age = datetime.now() - datetime.fromisoformat(cache_data['timestamp'])
                    if cache_age.total_seconds() < 86400:  # 24 hours
                        return cache_data.get('symbols', [])
            
            # Check if TradingView is available
            if not TRADINGVIEW_AVAILABLE:
                print("TradingView not available, using curated list")
                return self.get_curated_early_innings_list()
            
            # Use TradingView screener to get all US stocks $1B-$50B market cap
            print("Fetching fresh market data from TradingView...")
            
            query = (Query()
                    .select('ticker', 'market_cap_basic', 'volume', 'relative_volume_10d_calc')
                    .where(
                        col('market_cap_basic').between(1_000_000_000, 50_000_000_000),  # $1B to $50B
                        col('type').isin(['stock']),  # Only stocks, not ETFs
                        col('exchange').isin(['NASDAQ', 'NYSE', 'AMEX']),  # US exchanges
                        col('subtype').isin(['common', 'foreign-issuer']),  # Common stocks
                    )
                    .order_by('market_cap_basic', ascending=False)
                    .limit(500)  # Get top 500 by market cap
                    )
            
            # Execute the query
            data = query.get_scanner_data()
            
            # Extract tickers
            symbols = [row['ticker'] for row in data[1] if row.get('ticker')]
            
            # Save to cache
            cache_data = {
                'symbols': symbols,
                'timestamp': datetime.now().isoformat(),
                'count': len(symbols)
            }
            self.save_json(cache_file, cache_data)
            
            print(f"Found {len(symbols)} stocks in $1B-$50B range")
            return symbols
            
        except Exception as e:
            print(f"Error getting market universe: {e}")
            # Fall back to curated list if API fails
            return self.get_curated_early_innings_list()
    
    def get_curated_early_innings_list(self) -> List[str]:
        """Get expanded curated list of early-stage growth candidates"""
        # Expanded list covering more sectors and opportunities
        return [
            # AI/ML Infrastructure
            "AI", "BBAI", "SOUN", "GENI", "SPIR", "BIGB", "STEM", "LAZR",
            # Cybersecurity Next Gen
            "S", "TENB", "QLYS", "VRNS", "RPD", "JAMF", "PING", "MIME",
            # Cloud/SaaS Rising Stars  
            "MNDY", "BILL", "GTLB", "CFLT", "NCNO", "TOST", "ASAN", "SUMO",
            "DOCN", "ESTC", "FSLY", "FROG", "APPN", "YEXT", "ALRM", "NEWR",
            # Fintech Disruptors
            "AFRM", "UPST", "HOOD", "SOFI", "FLYW", "PAYO", "DAVE", "OPFI",
            "OPEN", "COMP", "PSFE", "VIRT", "STEP", "NRDS", "RELY", "FOUR",
            # Healthcare Innovation
            "DOCS", "ACCD", "RXRX", "SDGR", "PSTG", "ONEM", "HIMS", "TDOC",
            "VEEV", "PHR", "CERT", "OMCL", "PRVA", "PGNY", "EVH", "SGFY",
            # Gaming/Metaverse/Entertainment
            "RBLX", "U", "TTWO", "BMBL", "MGM", "DKNG", "PENN", "SKLZ",
            "GLBE", "SE", "HUYA", "DOYU", "YY", "FUBO", "VIAC", "LOGI",
            # Green Energy/EV/Clean Tech
            "RIVN", "LCID", "CHPT", "BLNK", "ENVX", "QS", "MVST", "GOEV",
            "FSR", "RIDE", "NKLA", "PTRA", "LEV", "ARVL", "SEV", "VLDR",
            "RUN", "NOVA", "CSIQ", "ARRY", "SPWR", "FSLR", "SEDG", "BE",
            # Space/Defense/Aerospace Tech
            "RKLB", "ASTR", "ASTS", "LUNR", "SPCE", "RDW", "MNTS", "SATL",
            # Biotech/Life Sciences
            "MRNA", "BNTX", "BEAM", "CRSP", "NTLA", "EDIT", "FATE", "SANA",
            "RVMD", "KRYS", "ARWR", "ALNY", "RARE", "IONS", "BMRN", "VRTX",
            # E-commerce/Digital Commerce
            "ETSY", "W", "POSH", "WISH", "FTCH", "REAL", "PRTS", "CVNA",
            "OPRT", "XMTR", "OSTK", "BFRI", "APRN", "WOOF", "PETS", "CHWY",
            # Data/Analytics
            "SNOW", "PLTR", "DDOG", "DT", "ESTC", "SPLK", "MDB", "CFLT",
            # Digital Media/Content
            "ROKU", "TTD", "MGNI", "PUBM", "APPS", "ZETA", "TBLA", "IAS"
        ]
    
    def get_recommended_positions(self, growth_tolerance: str = "balanced") -> Dict:
        """Get recommended positions based on screening results and growth tolerance"""
        latest_results = self.load_json(self.screening_results_file)
        
        if not latest_results or 'results' not in latest_results:
            return {"tier1": [], "tier2": [], "avoid_cc": []}
        
        recommendations = {
            "tier1": [],  # High conviction buys
            "tier2": [],  # Good opportunities
            "avoid_cc": []  # Don't sell covered calls on these
        }
        
        for result in latest_results['results']:
            score = result.get('score', 0)
            
            # High conviction plays
            if result['tier'] == 1 and score >= 80:
                recommendations['tier1'].append({
                    "symbol": result['symbol'],
                    "score": score,
                    "factors": result.get('factors', []),
                    "action": "BUILD POSITION - No covered calls"
                })
                recommendations['avoid_cc'].append(result['symbol'])
            
            # Good opportunities
            elif score >= 60:
                recommendations['tier2'].append({
                    "symbol": result['symbol'],
                    "score": score,
                    "factors": result.get('factors', []),
                    "action": "CONSIDER POSITION - Conservative CCs only"
                })
                if score >= 75:
                    recommendations['avoid_cc'].append(result['symbol'])
            
        return recommendations

# Example usage
if __name__ == "__main__":
    screener = GrowthScreeningSystem()
    
    # Run Tier 1 screen on sample symbols
    print("Running Tier 1 High Conviction Screen...")
    tier1_results = screener.screen_universe(["NVDA", "TSLA", "AAPL"], tier=1)
    
    # Show results
    for result in tier1_results:
        print(f"\n{result['symbol']} - Score: {result['score']}")
        print(f"Confidence: {result['confidence']}")
        print("Factors:", ", ".join(result['factors']))
    
    # Analyze effectiveness (would run periodically)
    print("\n\nAnalyzing screening effectiveness...")
    analysis = screener.analyze_and_refine()
    print(f"Success Rate: {analysis['success_rate']:.1%}")
    print("Recommendations:", analysis['recommended_adjustments'])