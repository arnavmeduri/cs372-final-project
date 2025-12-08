"""
Investment Score Calculator
Based on Benjamin Graham's criteria for defensive investors from "The Intelligent Investor" (1973).
Implements measurable criteria from Graham's 7-point framework for conservative stock selection.

Phase 4A: Enhanced with Phase 3A balance sheet integration and correct criterion implementations.
"""
from typing import Dict, Any, List, Optional
from .finnhub_client import FinancialMetrics
from .balance_sheet_analyzer import FinancialRatios


def calculate_investment_score(
    metrics: FinancialMetrics,
    balance_sheet_ratios: Optional[FinancialRatios] = None
) -> Dict[str, Any]:
    """
    Calculate investment score based on Benjamin Graham's defensive investor criteria.

    Based on "The Intelligent Investor" (1973), Chapter 14:
    Graham's 7 criteria for defensive investors:

    1. Adequate Size: Market cap > $2B (Graham's threshold, inflation-adjusted)
    2. Strong Financial Condition: Current Ratio ≥ 2.0 (Graham's actual criterion - Phase 4A corrected)
    3. Earnings Stability: Positive earnings in each of past 10 years (proxy: 1-year revenue growth)
    4. Dividend Record: Uninterrupted dividends for 20 years (proxy: current dividend yield > 0)
    5. Earnings Growth: EPS increase ≥ 33% over 10 years (proxy: revenue growth > 2.9% CAGR)
    6. Moderate P/E Ratio: P/E < 15 (Graham's value threshold)
    7. Moderate Price-to-Book: P/B < 1.5 or P/E × P/B < 22.5 (Phase 4A added)

    Phase 4A Improvements:
    - Integrated Phase 3A balance sheet ratios for correct Criterion 2
    - Added missing Criterion 7 (Price-to-Book)
    - Tracks data limitations and proxies used
    - Displays individual criterion results

    Reference: Graham, B. (1973). "The Intelligent Investor", 4th ed., Chapter 14.

    Args:
        metrics: FinancialMetrics object with company data
        balance_sheet_ratios: Optional FinancialRatios from Phase 3A balance sheet analysis

    Returns:
        Dictionary with:
        - score: 0-100 overall score (based on criteria met out of 7)
        - recommendation: BUY/HOLD/AVOID
        - signals: List of (type, message, criterion) tuples for each criterion
        - limitations: List of data quality warnings
        - methodology: Description of the scoring basis
    """
    criteria_met = 0
    total_criteria = 0
    signals = []
    limitations = []  # Track data quality issues

    # Criterion 1: Adequate Size (Graham: Market cap > $2B)
    if metrics.market_cap:
        total_criteria += 1
        # Market cap is in millions from Finnhub
        if metrics.market_cap >= 2000:  # $2B in millions
            criteria_met += 1
            signals.append(("positive", "Adequate size (Market cap > $2B)", "Criterion 1: Adequate Size"))
        else:
            signals.append(("negative", f"Insufficient size (Market cap ${metrics.market_cap/1000:.1f}B < $2B)", "Criterion 1: Adequate Size"))

    # Criterion 2: Strong Financial Condition (Graham: Current Ratio ≥ 2.0) - PHASE 4A CORRECTED
    if balance_sheet_ratios and balance_sheet_ratios.current_ratio is not None:
        total_criteria += 1
        cr = balance_sheet_ratios.current_ratio
        if cr >= 2.0:
            criteria_met += 1
            signals.append(("positive", f"Strong liquidity (Current Ratio {cr:.2f} ≥ 2.0)", "Criterion 2: Financial Condition"))
        elif cr >= 1.0:
            signals.append(("neutral", f"Adequate liquidity (Current Ratio {cr:.2f})", "Criterion 2: Financial Condition"))
        else:
            signals.append(("negative", f"Weak liquidity (Current Ratio {cr:.2f} < 1.0)", "Criterion 2: Financial Condition"))

        # Add context about working capital
        if balance_sheet_ratios.working_capital is not None:
            wc_billions = balance_sheet_ratios.working_capital / 1e9
            if wc_billions > 0:
                signals.append(("info", f"Positive working capital: ${wc_billions:.2f}B", "Additional Context"))
            else:
                signals.append(("warning", f"Negative working capital: ${wc_billions:.2f}B", "Additional Context"))
    elif metrics.debt_to_equity is not None:
        # Fallback to D/E if balance sheet not available (legacy behavior)
        total_criteria += 1
        limitations.append("⚠️ Balance sheet not available - using Debt-to-Equity as proxy for Criterion 2 (Graham requires Current Ratio ≥ 2.0)")
        if metrics.debt_to_equity < 1.0:
            criteria_met += 1
            signals.append(("positive", f"Conservative leverage (D/E {metrics.debt_to_equity:.2f} < 1.0)", "Criterion 2: Financial Condition (proxy)"))
        elif metrics.debt_to_equity < 1.5:
            signals.append(("neutral", f"Moderate leverage (D/E {metrics.debt_to_equity:.2f})", "Criterion 2: Financial Condition (proxy)"))
        else:
            signals.append(("negative", f"High leverage (D/E {metrics.debt_to_equity:.2f} > 1.5)", "Criterion 2: Financial Condition (proxy)"))
    
    # Criterion 3: Earnings Stability (Graham: Positive earnings each of past 10 years)
    # Proxy: 1-year revenue growth
    if metrics.revenue_growth_yoy is not None:
        total_criteria += 1
        limitations.append("⚠️ Earnings stability (Criterion 3) uses 1-year revenue as proxy - Graham requires positive earnings in each of past 10 years")
        if metrics.revenue_growth_yoy > 0:
            criteria_met += 1
            signals.append(("positive", f"Positive revenue growth ({metrics.revenue_growth_yoy:.1f}%)", "Criterion 3: Earnings Stability (proxy)"))
        else:
            signals.append(("negative", f"Declining revenue ({metrics.revenue_growth_yoy:.1f}%)", "Criterion 3: Earnings Stability (proxy)"))

    # Criterion 4: Dividend Record (Graham: Uninterrupted dividends for 20 years)
    # Proxy: Current dividend yield > 0
    if metrics.dividend_yield is not None:
        total_criteria += 1
        limitations.append("⚠️ Dividend record (Criterion 4) uses current yield - Graham requires uninterrupted dividends for at least 20 years")
        if metrics.dividend_yield > 0:
            criteria_met += 1
            signals.append(("positive", f"Dividend-paying (Yield {metrics.dividend_yield:.2f}%)", "Criterion 4: Dividend Record (proxy)"))
        else:
            signals.append(("negative", "No dividend yield", "Criterion 4: Dividend Record (proxy)"))

    # Criterion 5: Earnings Growth (Graham: EPS increase ≥ 33% over 10 years, ~2.9% CAGR)
    # Proxy: 1-year revenue growth > 2.9% (lowered from 5% in Phase 4A)
    if metrics.revenue_growth_yoy is not None:
        total_criteria += 1
        limitations.append("⚠️ Earnings growth (Criterion 5) uses 1-year revenue - Graham requires EPS growth of 33% over 10 years")
        if metrics.revenue_growth_yoy >= 3.0:  # Approximates 2.9% CAGR threshold
            criteria_met += 1
            signals.append(("positive", f"Solid growth ({metrics.revenue_growth_yoy:.1f}% ≥ 3.0%)", "Criterion 5: Earnings Growth (proxy)"))
        elif metrics.revenue_growth_yoy > 0:
            signals.append(("neutral", f"Moderate growth ({metrics.revenue_growth_yoy:.1f}%)", "Criterion 5: Earnings Growth (proxy)"))
        else:
            signals.append(("negative", f"No growth ({metrics.revenue_growth_yoy:.1f}%)", "Criterion 5: Earnings Growth (proxy)"))
    
    # Criterion 6: Moderate P/E Ratio (Graham: P/E < 15 for value)
    if metrics.pe_ratio:
        total_criteria += 1
        if metrics.pe_ratio < 15:
            criteria_met += 1
            signals.append(("positive", f"Attractive valuation (P/E {metrics.pe_ratio:.1f} < 15)", "Criterion 6: Moderate P/E Ratio"))
        elif metrics.pe_ratio < 25:
            signals.append(("neutral", f"Fair valuation (P/E {metrics.pe_ratio:.1f})", "Criterion 6: Moderate P/E Ratio"))
        else:
            signals.append(("negative", f"High valuation (P/E {metrics.pe_ratio:.1f} > 25)", "Criterion 6: Moderate P/E Ratio"))

    # Criterion 7: Moderate Price-to-Book (Graham: P/B < 1.5 or P/E × P/B < 22.5) - PHASE 4A ADDED
    # Need to calculate P/B from available data
    if hasattr(metrics, 'book_value_per_share') and hasattr(metrics, 'current_price'):
        if metrics.book_value_per_share and metrics.current_price and metrics.book_value_per_share > 0:
            total_criteria += 1
            pb_ratio = metrics.current_price / metrics.book_value_per_share

            if pb_ratio < 1.5:
                criteria_met += 1
                signals.append(("positive", f"Attractive P/B ratio ({pb_ratio:.2f} < 1.5)", "Criterion 7: Moderate Price-to-Book"))
            elif pb_ratio < 3.0:
                signals.append(("neutral", f"Fair P/B ratio ({pb_ratio:.2f})", "Criterion 7: Moderate Price-to-Book"))
            else:
                signals.append(("negative", f"High P/B ratio ({pb_ratio:.2f} > 3.0)", "Criterion 7: Moderate Price-to-Book"))

            # Graham's alternative criterion: P/E × P/B < 22.5
            if metrics.pe_ratio:
                combined_metric = metrics.pe_ratio * pb_ratio
                if combined_metric < 22.5:
                    signals.append(("info", f"Graham combined metric P/E×P/B = {combined_metric:.1f} < 22.5 ✓", "Alternative Valuation Test"))
                else:
                    signals.append(("info", f"Graham combined metric P/E×P/B = {combined_metric:.1f} > 22.5", "Alternative Valuation Test"))
    else:
        # P/B not available - note as limitation
        limitations.append("⚠️ Price-to-Book ratio (Criterion 7) not available - requires book value per share data")

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
        'limitations': limitations,  # Phase 4A: Data quality warnings
        'criteria_met': criteria_met,
        'total_criteria': total_criteria,
        'methodology': "Graham Defensive Investor Score (based on 'The Intelligent Investor', 1973) - Phase 4A Enhanced"
    }



