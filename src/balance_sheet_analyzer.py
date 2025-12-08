"""
Balance Sheet Analysis

Extracts financial data from balance sheets and calculates key ratios
for investment analysis.

Phase 3A: Enhanced Section Processing
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class FinancialRatios:
    """Container for calculated financial ratios."""
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    working_capital: Optional[float] = None
    cash_ratio: Optional[float] = None


class BalanceSheetAnalyzer:
    """
    Analyzes balance sheet data and calculates financial ratios.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize balance sheet analyzer.

        Args:
            verbose: Print extraction status
        """
        self.verbose = verbose

    def find_item(self, raw_data: List[Dict], concept_patterns: List[str]) -> Optional[float]:
        """
        Find a specific line item in balance sheet raw data.

        Args:
            raw_data: List of dicts from balance_sheet.get_raw_data()
            concept_patterns: List of concept name patterns to search for

        Returns:
            Most recent value (latest year) or None
        """
        for item in raw_data:
            concept = item.get('concept', '')
            label = item.get('label', '').lower()
            values = item.get('values', {})

            # Check if concept matches any pattern
            for pattern in concept_patterns:
                if pattern.lower() in concept.lower() or pattern.lower() in label:
                    # Get most recent value
                    if values:
                        # Sort by date (most recent first)
                        sorted_dates = sorted(values.keys(), reverse=True)
                        latest_value = values.get(sorted_dates[0])
                        if latest_value is not None:
                            return float(latest_value)

        return None

    def extract_balance_sheet_data(self, balance_sheet) -> Dict[str, Optional[float]]:
        """
        Extract key line items from balance sheet.

        Args:
            balance_sheet: edgartools Statement object

        Returns:
            Dictionary of line item values
        """
        try:
            raw_data = balance_sheet.get_raw_data()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to get balance sheet raw data: {e}")
            return {}

        # Extract key items with multiple possible concept names
        items = {
            'cash': self.find_item(raw_data, [
                'CashAndCashEquivalentsAtCarryingValue',
                'Cash',
                'CashAndCashEquivalents'
            ]),
            'marketable_securities': self.find_item(raw_data, [
                'MarketableSecurities',
                'ShortTermInvestments',
                'DebtSecuritiesAvailableForSale'
            ]),
            'current_assets': self.find_item(raw_data, [
                'AssetsCurrent',
                'CurrentAssets'
            ]),
            'inventory': self.find_item(raw_data, [
                'InventoryNet',
                'Inventory'
            ]),
            'current_liabilities': self.find_item(raw_data, [
                'LiabilitiesCurrent',
                'CurrentLiabilities'
            ]),
            'total_assets': self.find_item(raw_data, [
                'Assets',
                'TotalAssets'
            ]),
            'total_liabilities': self.find_item(raw_data, [
                'Liabilities',
                'TotalLiabilities'
            ]),
            'long_term_debt': self.find_item(raw_data, [
                'LongTermDebt',
                'LongTermDebtNoncurrent',
                'DebtLongTerm'
            ]),
            'short_term_debt': self.find_item(raw_data, [
                'ShortTermDebt',
                'DebtCurrent',
                'ShortTermBorrowings'
            ]),
            'stockholders_equity': self.find_item(raw_data, [
                'StockholdersEquity',
                'ShareholdersEquity',
                'Equity'
            ]),
        }

        if self.verbose:
            found = sum(1 for v in items.values() if v is not None)
            print(f"✅ Extracted {found}/{len(items)} balance sheet line items")

        return items

    def calculate_ratios(self, items: Dict[str, Optional[float]]) -> FinancialRatios:
        """
        Calculate financial ratios from balance sheet items.

        Args:
            items: Dictionary of balance sheet line items

        Returns:
            FinancialRatios object
        """
        ratios = FinancialRatios()

        # Current Ratio = Current Assets / Current Liabilities
        if items.get('current_assets') and items.get('current_liabilities'):
            ca = items['current_assets']
            cl = items['current_liabilities']
            if cl != 0:
                ratios.current_ratio = ca / cl

        # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        if items.get('current_assets') and items.get('current_liabilities') and items.get('inventory'):
            ca = items['current_assets']
            inv = items['inventory']
            cl = items['current_liabilities']
            if cl != 0:
                ratios.quick_ratio = (ca - inv) / cl

        # Working Capital = Current Assets - Current Liabilities
        if items.get('current_assets') and items.get('current_liabilities'):
            ratios.working_capital = items['current_assets'] - items['current_liabilities']

        # Debt-to-Equity = Total Debt / Stockholders' Equity
        if items.get('stockholders_equity'):
            equity = items['stockholders_equity']
            total_debt = 0

            if items.get('long_term_debt'):
                total_debt += items['long_term_debt']
            if items.get('short_term_debt'):
                total_debt += items['short_term_debt']

            if equity != 0 and total_debt > 0:
                ratios.debt_to_equity = total_debt / equity

        # Cash Ratio = (Cash + Marketable Securities) / Current Liabilities
        if items.get('current_liabilities'):
            cl = items['current_liabilities']
            liquid_assets = 0

            if items.get('cash'):
                liquid_assets += items['cash']
            if items.get('marketable_securities'):
                liquid_assets += items['marketable_securities']

            if cl != 0 and liquid_assets > 0:
                ratios.cash_ratio = liquid_assets / cl

        return ratios

    def interpret_ratios(self, ratios: FinancialRatios) -> str:
        """
        Provide beginner-friendly interpretation of financial ratios.

        Args:
            ratios: FinancialRatios object

        Returns:
            Human-readable interpretation
        """
        insights = []

        # Current Ratio interpretation
        if ratios.current_ratio is not None:
            cr = ratios.current_ratio
            if cr < 1.0:
                insights.append(
                    f"⚠️  Current Ratio: {cr:.2f} - Below 1.0 suggests potential liquidity concerns. "
                    f"The company may have difficulty paying short-term debts."
                )
            elif cr > 2.0:
                insights.append(
                    f"✅ Current Ratio: {cr:.2f} - Strong liquidity position. "
                    f"The company has ample current assets to cover short-term obligations."
                )
            else:
                insights.append(
                    f"Current Ratio: {cr:.2f} - Adequate liquidity. "
                    f"The company can cover current liabilities with current assets."
                )

        # Quick Ratio interpretation
        if ratios.quick_ratio is not None:
            qr = ratios.quick_ratio
            if qr < 0.5:
                insights.append(
                    f"⚠️  Quick Ratio: {qr:.2f} - Low quick liquidity. "
                    f"May struggle to meet immediate obligations without selling inventory."
                )
            elif qr >= 1.0:
                insights.append(
                    f"✅ Quick Ratio: {qr:.2f} - Can cover short-term liabilities "
                    f"without relying on inventory sales."
                )
            else:
                insights.append(
                    f"Quick Ratio: {qr:.2f} - Moderate quick liquidity."
                )

        # Debt-to-Equity interpretation
        if ratios.debt_to_equity is not None:
            dte = ratios.debt_to_equity
            if dte > 2.0:
                insights.append(
                    f"⚠️  Debt-to-Equity: {dte:.2f} - High financial leverage. "
                    f"Significant debt relative to equity increases financial risk."
                )
            elif dte < 0.5:
                insights.append(
                    f"✅ Debt-to-Equity: {dte:.2f} - Conservative capital structure. "
                    f"Low reliance on debt financing."
                )
            else:
                insights.append(
                    f"Debt-to-Equity: {dte:.2f} - Moderate leverage. "
                    f"Balanced use of debt and equity financing."
                )

        # Working Capital interpretation
        if ratios.working_capital is not None:
            wc = ratios.working_capital
            wc_billions = wc / 1e9
            if wc < 0:
                insights.append(
                    f"⚠️  Working Capital: ${wc_billions:.2f}B - Negative working capital "
                    f"indicates current liabilities exceed current assets."
                )
            else:
                insights.append(
                    f"Working Capital: ${wc_billions:.2f}B - Positive working capital "
                    f"provides financial flexibility."
                )

        # Cash Ratio interpretation
        if ratios.cash_ratio is not None:
            cash_r = ratios.cash_ratio
            if cash_r < 0.2:
                insights.append(
                    f"Cash Ratio: {cash_r:.2f} - Limited immediate liquidity. "
                    f"Less than 20% of current liabilities covered by cash."
                )
            elif cash_r > 0.5:
                insights.append(
                    f"✅ Cash Ratio: {cash_r:.2f} - Strong cash position. "
                    f"Over 50% of current liabilities can be covered with liquid assets."
                )

        return "\n\n".join(insights) if insights else "Insufficient data for ratio analysis."

    def format_for_context(
        self,
        items: Dict[str, Optional[float]],
        ratios: FinancialRatios,
        company_name: str
    ) -> str:
        """
        Format balance sheet analysis for inclusion in LLM context.
        De-emphasized formatting to avoid overwhelming other sections.

        Args:
            items: Balance sheet line items
            ratios: Calculated ratios
            company_name: Company name

        Returns:
            Formatted text for context
        """
        output = []
        # De-emphasized header (no prominent separators to avoid over-focus)
        output.append(f"Balance Sheet Analysis for {company_name}:")

        # Key line items (in billions)
        output.append("Key Balance Sheet Items:")
        if items.get('total_assets'):
            output.append(f"  Total Assets: ${items['total_assets'] / 1e9:.2f}B")
        if items.get('current_assets'):
            output.append(f"  Current Assets: ${items['current_assets'] / 1e9:.2f}B")
        if items.get('cash'):
            output.append(f"  Cash: ${items['cash'] / 1e9:.2f}B")
        if items.get('current_liabilities'):
            output.append(f"  Current Liabilities: ${items['current_liabilities'] / 1e9:.2f}B")
        if items.get('stockholders_equity'):
            output.append(f"  Stockholders' Equity: ${items['stockholders_equity'] / 1e9:.2f}B")

        output.append("")

        # Calculated ratios
        output.append("Calculated Financial Ratios:")
        if ratios.current_ratio is not None:
            output.append(f"  Current Ratio: {ratios.current_ratio:.2f}")
        if ratios.quick_ratio is not None:
            output.append(f"  Quick Ratio: {ratios.quick_ratio:.2f}")
        if ratios.debt_to_equity is not None:
            output.append(f"  Debt-to-Equity: {ratios.debt_to_equity:.2f}")
        if ratios.working_capital is not None:
            output.append(f"  Working Capital: ${ratios.working_capital / 1e9:.2f}B")
        if ratios.cash_ratio is not None:
            output.append(f"  Cash Ratio: {ratios.cash_ratio:.2f}")

        output.append("")

        # Interpretation
        output.append("Interpretation:")
        interpretation = self.interpret_ratios(ratios)
        output.append(interpretation)

        return "\n".join(output)

    def analyze(self, balance_sheet, company_name: str) -> Dict[str, Any]:
        """
        Complete analysis of balance sheet.

        Args:
            balance_sheet: edgartools Statement object
            company_name: Company name

        Returns:
            Dictionary with items, ratios, interpretation, and formatted_context
        """
        # Extract items
        items = self.extract_balance_sheet_data(balance_sheet)

        # Calculate ratios
        ratios = self.calculate_ratios(items)

        # Generate interpretation
        interpretation = self.interpret_ratios(ratios)

        # Format for context
        formatted_context = self.format_for_context(items, ratios, company_name)

        return {
            'items': items,
            'ratios': ratios,
            'interpretation': interpretation,
            'formatted_context': formatted_context,
            'has_data': any(v is not None for v in items.values())
        }
