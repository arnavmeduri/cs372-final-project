"""
Investment Score Calculator
Based on Benjamin Graham's criteria for defensive investors from "The Intelligent Investor" (1973).
Implements measurable criteria from Graham's 7-point framework for conservative stock selection.
"""
from typing import Dict, Any, List
from .finnhub_client import FinancialMetrics


def calculate_investment_score(metrics: FinancialMetrics) -> Dict[str, Any]:
    """
    Calculate investment score based on Benjamin Graham's defensive investor criteria.
    
    Based on "The Intelligent Investor" (1973), Chapter 14:
    Graham's 7 criteria for defensive investors (implementing those measurable with available data):
    
    1. Adequate Size: Market cap > $2B (Graham's threshold)
    2. Strong Financial Condition: Debt-to-equity < 1.0 (Graham's conservative threshold)
    3. Earnings Stability: Positive revenue growth (proxy for earnings stability)
    4. Dividend Record: Dividend yield > 0% (indicates dividend-paying history)
    5. Earnings Growth: Revenue growth > 0% (proxy for earnings growth)
    6. Moderate P/E Ratio: P/E < 15 (Graham's value threshold)
    7. Moderate Price-to-Book: Not available in current data
    
    Reference: Graham, B. (1973). "The Intelligent Investor", 4th ed., Chapter 14.
    
    Args:
        metrics: FinancialMetrics object with company data
        
    Returns:
        Dictionary with:
        - score: 0-100 overall score (based on criteria met)
        - recommendation: BUY/HOLD/AVOID
        - signals: List of contributing factors with Graham criteria references
        - methodology: Description of the scoring basis
    """
    criteria_met = 0
    total_criteria = 0
    signals = []
    
    # Criterion 1: Adequate Size (Graham: Market cap > $2B)
    if metrics.market_cap:
        total_criteria += 1
        # Market cap is in millions from Finnhub
        if metrics.market_cap >= 2000:  # $2B in millions
            criteria_met += 1
            signals.append(("positive", "Adequate size (Market cap > $2B)", "Graham Criterion 1"))
        else:
            signals.append(("negative", f"Insufficient size (Market cap ${metrics.market_cap/1000:.1f}B < $2B)", "Graham Criterion 1"))
    
    # Criterion 2: Strong Financial Condition (Graham: D/E < 1.0 for conservative)
    if metrics.debt_to_equity is not None:
        total_criteria += 1
        if metrics.debt_to_equity < 1.0:
            criteria_met += 1
            signals.append(("positive", f"Strong financial condition (D/E {metrics.debt_to_equity:.2f} < 1.0)", "Graham Criterion 2"))
        elif metrics.debt_to_equity < 1.5:
            signals.append(("neutral", f"Moderate leverage (D/E {metrics.debt_to_equity:.2f})", "Graham Criterion 2"))
        else:
            signals.append(("negative", f"High leverage (D/E {metrics.debt_to_equity:.2f} > 1.5)", "Graham Criterion 2"))
    
    # Criterion 3: Earnings Stability (Proxy: Positive revenue growth)
    if metrics.revenue_growth_yoy is not None:
        total_criteria += 1
        if metrics.revenue_growth_yoy > 0:
            criteria_met += 1
            signals.append(("positive", f"Positive revenue growth ({metrics.revenue_growth_yoy:.1f}%)", "Graham Criterion 3 (proxy)"))
        else:
            signals.append(("negative", f"Declining revenue ({metrics.revenue_growth_yoy:.1f}%)", "Graham Criterion 3 (proxy)"))
    
    # Criterion 4: Dividend Record (Graham: Consistent dividend payments)
    if metrics.dividend_yield is not None:
        total_criteria += 1
        if metrics.dividend_yield > 0:
            criteria_met += 1
            signals.append(("positive", f"Dividend-paying (Yield {metrics.dividend_yield:.2f}%)", "Graham Criterion 4"))
        else:
            signals.append(("negative", "No dividend yield", "Graham Criterion 4"))
    
    # Criterion 5: Earnings Growth (Proxy: Revenue growth > 5%)
    if metrics.revenue_growth_yoy is not None:
        total_criteria += 1
        if metrics.revenue_growth_yoy > 5:
            criteria_met += 1
            signals.append(("positive", f"Strong growth ({metrics.revenue_growth_yoy:.1f}% > 5%)", "Graham Criterion 5 (proxy)"))
        elif metrics.revenue_growth_yoy > 0:
            signals.append(("neutral", f"Moderate growth ({metrics.revenue_growth_yoy:.1f}%)", "Graham Criterion 5 (proxy)"))
        else:
            signals.append(("negative", f"No growth ({metrics.revenue_growth_yoy:.1f}%)", "Graham Criterion 5 (proxy)"))
    
    # Criterion 6: Moderate P/E Ratio (Graham: P/E < 15 for value)
    if metrics.pe_ratio:
        total_criteria += 1
        if metrics.pe_ratio < 15:
            criteria_met += 1
            signals.append(("positive", f"Moderate P/E ratio ({metrics.pe_ratio:.1f} < 15)", "Graham Criterion 6"))
        elif metrics.pe_ratio < 25:
            signals.append(("neutral", f"Moderate P/E ratio ({metrics.pe_ratio:.1f})", "Graham Criterion 6"))
        else:
            signals.append(("negative", f"High P/E ratio ({metrics.pe_ratio:.1f} > 25)", "Graham Criterion 6"))
    
    # Calculate score as percentage of criteria met
    if total_criteria > 0:
        score = int((criteria_met / total_criteria) * 100)
    else:
        score = 50  # Default if no data
    
    # Convert to recommendation based on Graham's conservative approach
    if score >= 80:  # 5+ out of 6 criteria
        recommendation = "BUY"
        rec_explanation = "Meets most Graham defensive investor criteria"
    elif score >= 50:  # 3-4 out of 6 criteria
        recommendation = "HOLD"
        rec_explanation = "Meets some Graham defensive investor criteria"
    else:  # < 3 out of 6 criteria
        recommendation = "AVOID"
        rec_explanation = "Fails to meet most Graham defensive investor criteria"
    
    return {
        'score': score,
        'recommendation': recommendation,
        'recommendation_explanation': rec_explanation,
        'signals': signals,
        'criteria_met': criteria_met,
        'total_criteria': total_criteria,
        'methodology': "Graham Defensive Investor Score (based on 'The Intelligent Investor', 1973)"
    }



