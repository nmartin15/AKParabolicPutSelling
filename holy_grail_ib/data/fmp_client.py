"""
Financial Modeling Prep API Client for Holy Grail Options Strategy
Handles fundamental data collection for business quality screening and DCF valuation
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import json
from urllib.parse import urljoin

from config.criteria import get_criteria
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FinancialData:
    """Structure for financial statement data"""
    symbol: str
    year: int
    quarter: Optional[int]
    revenue: float
    net_income: float
    free_cash_flow: float
    total_debt: float
    cash_and_equivalents: float
    total_assets: float
    total_equity: float
    shares_outstanding: float
    operating_cash_flow: float
    capital_expenditures: float
    operating_income: float
    gross_profit: float
    cost_of_revenue: float


@dataclass
class RatiosData:
    """Structure for financial ratios"""
    symbol: str
    year: int
    roe: float
    roic: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    asset_turnover: float
    days_sales_outstanding: float
    free_cash_flow_yield: float
    revenue_growth: float
    net_income_growth: float


@dataclass
class CompanyProfile:
    """Structure for company profile data"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    description: str
    market_cap: float
    beta: float
    employees: int
    country: str
    exchange: str
    website: str
    ceo: str


class FMPClient:
    """
    Financial Modeling Prep API client for fundamental data.
    Provides comprehensive financial data for business quality screening.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://financialmodelingprep.com/api/v3"):
        """
        Initialize FMP client.
        
        Args:
            api_key: FMP API key
            base_url: FMP API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
        # Rate limiting (300 requests per minute for paid plans)
        self.rate_limit_delay = 0.25  # 250ms between requests
        self.last_request_time = 0
        
        # Cache for expensive calls
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make rate-limited API request to FMP.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response data or None if error
        """
        try:
            # Rate limiting
            now = datetime.now().timestamp()
            time_since_last = now - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            # Prepare request
            url = urljoin(self.base_url, endpoint)
            request_params = params or {}
            request_params['apikey'] = self.api_key
            
            # Check cache first
            cache_key = f"{endpoint}_{json.dumps(request_params, sort_keys=True)}"
            if cache_key in self._cache:
                if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                    logger.debug(f"Cache hit for {endpoint}")
                    return self._cache[cache_key]
            
            # Make request
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, params=request_params) as response:
                self.last_request_time = datetime.now().timestamp()
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful responses for 1 hour
                    self._cache[cache_key] = data
                    self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                    
                    return data
                else:
                    logger.error(f"FMP API error {response.status} for {endpoint}: {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error making FMP request to {endpoint}: {e}")
            return None
    
    async def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """
        Get company profile information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            CompanyProfile object or None
        """
        try:
            data = await self._make_request(f"profile/{symbol}")
            
            if data and len(data) > 0:
                profile_data = data[0]
                
                return CompanyProfile(
                    symbol=symbol,
                    company_name=profile_data.get('companyName', ''),
                    sector=profile_data.get('sector', ''),
                    industry=profile_data.get('industry', ''),
                    description=profile_data.get('description', ''),
                    market_cap=profile_data.get('mktCap', 0),
                    beta=profile_data.get('beta', 0),
                    employees=profile_data.get('fullTimeEmployees', 0),
                    country=profile_data.get('country', ''),
                    exchange=profile_data.get('exchangeShortName', ''),
                    website=profile_data.get('website', ''),
                    ceo=profile_data.get('ceo', '')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting company profile for {symbol}: {e}")
            return None
    
    async def get_income_statements(self, symbol: str, limit: int = 10) -> List[FinancialData]:
        """
        Get income statement data for multiple years.
        
        Args:
            symbol: Stock symbol
            limit: Number of years to retrieve
            
        Returns:
            List of FinancialData objects
        """
        try:
            data = await self._make_request(f"income-statement/{symbol}", {'limit': limit})
            
            if not data:
                return []
            
            financial_data = []
            for item in data:
                try:
                    financial_data.append(FinancialData(
                        symbol=symbol,
                        year=int(item.get('calendarYear', 0)),
                        quarter=None,  # Annual data
                        revenue=float(item.get('revenue', 0)),
                        net_income=float(item.get('netIncome', 0)),
                        free_cash_flow=0,  # Will get from cash flow statement
                        total_debt=0,      # Will get from balance sheet
                        cash_and_equivalents=0,  # Will get from balance sheet
                        total_assets=0,    # Will get from balance sheet
                        total_equity=0,    # Will get from balance sheet
                        shares_outstanding=float(item.get('weightedAverageShsOut', 0)),
                        operating_cash_flow=0,  # Will get from cash flow statement
                        capital_expenditures=0, # Will get from cash flow statement
                        operating_income=float(item.get('operatingIncome', 0)),
                        gross_profit=float(item.get('grossProfit', 0)),
                        cost_of_revenue=float(item.get('costOfRevenue', 0))
                    ))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing income statement data for {symbol} year {item.get('calendarYear')}: {e}")
                    continue
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error getting income statements for {symbol}: {e}")
            return []
    
    async def get_balance_sheets(self, symbol: str, limit: int = 10) -> Dict[int, Dict]:
        """
        Get balance sheet data for multiple years.
        
        Args:
            symbol: Stock symbol
            limit: Number of years to retrieve
            
        Returns:
            Dict mapping year to balance sheet data
        """
        try:
            data = await self._make_request(f"balance-sheet-statement/{symbol}", {'limit': limit})
            
            if not data:
                return {}
            
            balance_sheets = {}
            for item in data:
                try:
                    year = int(item.get('calendarYear', 0))
                    balance_sheets[year] = {
                        'total_debt': float(item.get('totalDebt', 0)),
                        'cash_and_equivalents': float(item.get('cashAndCashEquivalents', 0)),
                        'total_assets': float(item.get('totalAssets', 0)),
                        'total_equity': float(item.get('totalStockholdersEquity', 0)),
                        'current_assets': float(item.get('totalCurrentAssets', 0)),
                        'current_liabilities': float(item.get('totalCurrentLiabilities', 0)),
                        'inventory': float(item.get('inventory', 0)),
                        'accounts_receivable': float(item.get('netReceivables', 0))
                    }
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing balance sheet data for {symbol} year {item.get('calendarYear')}: {e}")
                    continue
            
            return balance_sheets
            
        except Exception as e:
            logger.error(f"Error getting balance sheets for {symbol}: {e}")
            return {}
    
    async def get_cash_flow_statements(self, symbol: str, limit: int = 10) -> Dict[int, Dict]:
        """
        Get cash flow statement data for multiple years.
        
        Args:
            symbol: Stock symbol
            limit: Number of years to retrieve
            
        Returns:
            Dict mapping year to cash flow data
        """
        try:
            data = await self._make_request(f"cash-flow-statement/{symbol}", {'limit': limit})
            
            if not data:
                return {}
            
            cash_flows = {}
            for item in data:
                try:
                    year = int(item.get('calendarYear', 0))
                    cash_flows[year] = {
                        'operating_cash_flow': float(item.get('netCashProvidedByOperatingActivities', 0)),
                        'capital_expenditures': float(item.get('capitalExpenditure', 0)),
                        'free_cash_flow': float(item.get('freeCashFlow', 0)),
                        'depreciation': float(item.get('depreciationAndAmortization', 0))
                    }
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing cash flow data for {symbol} year {item.get('calendarYear')}: {e}")
                    continue
            
            return cash_flows
            
        except Exception as e:
            logger.error(f"Error getting cash flow statements for {symbol}: {e}")
            return {}
    
    async def get_financial_ratios(self, symbol: str, limit: int = 10) -> List[RatiosData]:
        """
        Get financial ratios for multiple years.
        
        Args:
            symbol: Stock symbol
            limit: Number of years to retrieve
            
        Returns:
            List of RatiosData objects
        """
        try:
            data = await self._make_request(f"ratios/{symbol}", {'limit': limit})
            
            if not data:
                return []
            
            ratios_data = []
            for item in data:
                try:
                    ratios_data.append(RatiosData(
                        symbol=symbol,
                        year=int(item.get('calendarYear', 0)),
                        roe=float(item.get('returnOnEquity', 0)),
                        roic=float(item.get('returnOnCapitalEmployed', 0)),  # Closest to ROIC
                        debt_to_equity=float(item.get('debtEquityRatio', 0)),
                        current_ratio=float(item.get('currentRatio', 0)),
                        quick_ratio=float(item.get('quickRatio', 0)),
                        gross_margin=float(item.get('grossProfitMargin', 0)),
                        operating_margin=float(item.get('operatingProfitMargin', 0)),
                        net_margin=float(item.get('netProfitMargin', 0)),
                        asset_turnover=float(item.get('assetTurnover', 0)),
                        days_sales_outstanding=float(item.get('daysOfSalesOutstanding', 0)),
                        free_cash_flow_yield=float(item.get('freeCashFlowYield', 0)),
                        revenue_growth=float(item.get('revenueGrowth', 0)),
                        net_income_growth=float(item.get('netIncomeGrowth', 0))
                    ))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing ratios data for {symbol} year {item.get('calendarYear')}: {e}")
                    continue
            
            return ratios_data
            
        except Exception as e:
            logger.error(f"Error getting financial ratios for {symbol}: {e}")
            return []
    
    async def get_complete_financial_data(self, symbol: str, years: int = 10) -> Optional[Dict]:
        """
        Get complete financial data package for a company.
        Combines income statement, balance sheet, cash flow, and ratios.
        
        Args:
            symbol: Stock symbol
            years: Number of years of data to retrieve
            
        Returns:
            Dict with complete financial data or None
        """
        try:
            logger.info(f"Fetching complete financial data for {symbol}")
            
            # Get all financial statements concurrently
            tasks = [
                self.get_company_profile(symbol),
                self.get_income_statements(symbol, years),
                self.get_balance_sheets(symbol, years),
                self.get_cash_flow_statements(symbol, years),
                self.get_financial_ratios(symbol, years)
            ]
            
            profile, income_data, balance_data, cashflow_data, ratios_data = await asyncio.gather(*tasks)
            
            if not income_data:
                logger.warning(f"No income statement data found for {symbol}")
                return None
            
            # Combine all data by year
            complete_data = {
                'profile': profile,
                'years_data': {},
                'ratios': {ratio.year: ratio for ratio in ratios_data}
            }
            
            # Merge financial statements by year
            for income in income_data:
                year = income.year
                complete_data['years_data'][year] = {
                    'income': income,
                    'balance': balance_data.get(year, {}),
                    'cashflow': cashflow_data.get(year, {})
                }
            
            logger.info(f"Retrieved {len(complete_data['years_data'])} years of data for {symbol}")
            return complete_data
            
        except Exception as e:
            logger.error(f"Error getting complete financial data for {symbol}: {e}")
            return None
    
    async def get_analyst_estimates(self, symbol: str) -> Optional[Dict]:
        """
        Get analyst estimates for revenue and earnings growth.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with analyst estimates
        """
        try:
            data = await self._make_request(f"analyst-estimates/{symbol}")
            
            if not data:
                return None
            
            estimates = {}
            for item in data:
                year = item.get('date', '')[:4]  # Extract year from date
                estimates[year] = {
                    'estimated_revenue': float(item.get('estimatedRevenueAvg', 0)),
                    'estimated_ebitda': float(item.get('estimatedEbitdaAvg', 0)),
                    'estimated_eps': float(item.get('estimatedEpsAvg', 0)),
                    'estimated_revenue_high': float(item.get('estimatedRevenueHigh', 0)),
                    'estimated_revenue_low': float(item.get('estimatedRevenueLow', 0))
                }
            
            return estimates
            
        except Exception as e:
            logger.error(f"Error getting analyst estimates for {symbol}: {e}")
            return None
    
    async def get_dcf_inputs(self, symbol: str) -> Optional[Dict]:
        """
        Get all necessary inputs for DCF valuation calculation.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with DCF calculation inputs
        """
        try:
            # Get complete financial data
            financial_data = await self.get_complete_financial_data(symbol)
            if not financial_data:
                return None
            
            # Get analyst estimates for growth projections
            estimates = await self.get_analyst_estimates(symbol)
            
            # Calculate historical growth rates
            years_data = financial_data['years_data']
            sorted_years = sorted(years_data.keys(), reverse=True)
            
            if len(sorted_years) < 5:
                logger.warning(f"Insufficient data for DCF calculation: {symbol}")
                return None
            
            # Calculate 5-year revenue CAGR
            latest_year = sorted_years[0]
            five_years_ago = sorted_years[4] if len(sorted_years) >= 5 else sorted_years[-1]
            
            latest_revenue = years_data[latest_year]['income'].revenue
            old_revenue = years_data[five_years_ago]['income'].revenue
            
            if old_revenue > 0:
                revenue_cagr = ((latest_revenue / old_revenue) ** (1/5)) - 1
            else:
                revenue_cagr = 0
            
            # Get latest financial metrics
            latest_data = years_data[latest_year]
            latest_ratios = financial_data['ratios'].get(latest_year)
            
            dcf_inputs = {
                'symbol': symbol,
                'current_revenue': latest_revenue,
                'current_fcf': latest_data['cashflow'].get('free_cash_flow', 0),
                'current_shares': latest_data['income'].shares_outstanding,
                'historical_revenue_cagr': revenue_cagr,
                'current_operating_margin': latest_ratios.operating_margin if latest_ratios else 0,
                'current_tax_rate': 0.21,  # Standard corporate tax rate
                'beta': financial_data['profile'].beta if financial_data['profile'] else 1.0,
                'market_cap': financial_data['profile'].market_cap if financial_data['profile'] else 0,
                'analyst_estimates': estimates,
                'financial_data': financial_data
            }
            
            return dcf_inputs
            
        except Exception as e:
            logger.error(f"Error getting DCF inputs for {symbol}: {e}")
            return None
    
    async def screen_sp500_universe(self) -> List[str]:
        """
        Get list of S&P 500 companies for screening.
        
        Returns:
            List of S&P 500 symbols
        """
        try:
            data = await self._make_request("sp500_constituent")
            
            if data:
                symbols = [item.get('symbol') for item in data if item.get('symbol')]
                logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
                return symbols
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 universe: {e}")
            return []
    
    async def batch_company_profiles(self, symbols: List[str]) -> Dict[str, CompanyProfile]:
        """
        Get company profiles for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to CompanyProfile
        """
        profiles = {}
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = [self.get_company_profile(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, CompanyProfile):
                    profiles[symbol] = result
                elif isinstance(result, Exception):
                    logger.debug(f"Error getting profile for {symbol}: {result}")
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        logger.info(f"Retrieved profiles for {len(profiles)} companies")
        return profiles


# Singleton instance for global use
fmp_client = None

async def get_fmp_client(api_key: str) -> FMPClient:
    """
    Get or create FMP client singleton.
    
    Args:
        api_key: FMP API key
        
    Returns:
        FMPClient instance
    """
    global fmp_client
    
    if fmp_client is None:
        fmp_client = FMPClient(api_key)
    
    return fmp_client