"""
Business Quality Screener for Holy Grail Options Strategy
Implements Adam Khoo's 5-filter business quality screening system

Filters out 95% of companies, leaving only the highest quality businesses
that meet all quantified criteria for consistent growth and financial strength.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statistics

from config.criteria import get_criteria, PreferredSector, ExcludedSector
from data.fmp_client import get_fmp_client, FinancialData, RatiosData, CompanyProfile
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class QualityScore:
    """Business quality scoring results"""
    symbol: str
    total_score: int
    max_score: int
    passed: bool
    
    # Individual filter scores
    revenue_growth_score: int
    profitability_score: int
    balance_sheet_score: int
    moat_strength_score: int
    management_score: int
    
    # Detailed breakdown
    filter_results: Dict[str, Dict[str, Any]]
    disqualification_reasons: List[str]


@dataclass
class QualifiedCompany:
    """Company that passed all quality filters"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    quality_score: QualityScore
    financial_summary: Dict[str, float]


class BusinessQualityScreener:
    """
    Implements Adam Khoo's 5-filter business quality screening system.
    
    Filter 1: Revenue Growth Consistency (20 points)
    Filter 2: Profitability & Cash Flow Strength (15 points)  
    Filter 3: Balance Sheet Requirements (20 points)
    Filter 4: Competitive Moat Metrics (25 points)
    Filter 5: Management Quality Metrics (20 points)
    
    Must score ≥80/100 to qualify for options trading.
    """
    
    def __init__(self, fmp_api_key: str):
        """
        Initialize business quality screener.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key
        """
        self.fmp_api_key = fmp_api_key
        self.criteria = get_criteria('business_quality')
        self.sector_criteria = get_criteria('sector_scoring')
        self.scoring_weights = get_criteria('scoring_weights')
        
        # Sector preferences
        self.preferred_sectors = {sector.value for sector in PreferredSector}
        self.excluded_sectors = {sector.value for sector in ExcludedSector}
        
    async def screen_universe(self, symbols: List[str] = None) -> List[QualifiedCompany]:
        """
        Screen entire universe of stocks for business quality.
        
        Args:
            symbols: List of symbols to screen, or None for S&P 500
            
        Returns:
            List of companies that passed all quality filters
        """
        try:
            async with await get_fmp_client(self.fmp_api_key) as fmp:
                # Get universe to screen
                if symbols is None:
                    logger.info("Screening S&P 500 universe")
                    symbols = await fmp.screen_sp500_universe()
                else:
                    logger.info(f"Screening {len(symbols)} provided symbols")
                
                if not symbols:
                    logger.error("No symbols to screen")
                    return []
                
                # First pass: sector filtering
                logger.info("Step 1: Sector filtering")
                sector_qualified = await self._sector_filter(fmp, symbols)
                logger.info(f"Sector filter: {len(sector_qualified)}/{len(symbols)} passed")
                
                # Second pass: detailed financial screening
                logger.info("Step 2: Financial quality screening")
                qualified_companies = []
                
                # Process in batches to manage API rate limits
                batch_size = 20
                for i in range(0, len(sector_qualified), batch_size):
                    batch = sector_qualified[i:i + batch_size]
                    batch_results = await self._screen_batch(fmp, batch)
                    qualified_companies.extend(batch_results)
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_results)} qualified")
                    await asyncio.sleep(1)  # Rate limiting
                
                logger.info(f"Final results: {len(qualified_companies)}/{len(symbols)} companies passed all filters")
                
                # Sort by quality score
                qualified_companies.sort(key=lambda x: x.quality_score.total_score, reverse=True)
                
                return qualified_companies
                
        except Exception as e:
            logger.error(f"Error screening universe: {e}")
            return []
    
    async def _sector_filter(self, fmp, symbols: List[str]) -> List[str]:
        """
        Filter companies by sector preferences.
        
        Args:
            fmp: FMP client instance
            symbols: List of symbols to filter
            
        Returns:
            List of symbols in preferred sectors
        """
        try:
            # Get company profiles in batches
            profiles = await fmp.batch_company_profiles(symbols)
            
            sector_qualified = []
            for symbol, profile in profiles.items():
                if profile and self._is_preferred_sector(profile.sector):
                    sector_qualified.append(symbol)
                elif profile and profile.sector in self.excluded_sectors:
                    logger.debug(f"{symbol} excluded: sector {profile.sector}")
                else:
                    logger.debug(f"{symbol} neutral sector: {profile.sector if profile else 'Unknown'}")
            
            return sector_qualified
            
        except Exception as e:
            logger.error(f"Error in sector filtering: {e}")
            return symbols  # Return all if sector filtering fails
    
    def _is_preferred_sector(self, sector: str) -> bool:
        """Check if sector is in preferred list"""
        if not sector:
            return False
        
        # Map common sector names to our preferred categories
        sector_mappings = {
            'Technology': PreferredSector.TECHNOLOGY.value,
            'Healthcare': PreferredSector.HEALTHCARE.value,
            'Consumer Cyclical': PreferredSector.CONSUMER_DISCRETIONARY.value,
            'Consumer Discretionary': PreferredSector.CONSUMER_DISCRETIONARY.value,
            'Communication Services': PreferredSector.COMMUNICATION_SERVICES.value,
            'Financial Services': PreferredSector.SELECT_FINANCIALS.value,
            'Financials': PreferredSector.SELECT_FINANCIALS.value
        }
        
        mapped_sector = sector_mappings.get(sector, sector)
        return mapped_sector in self.preferred_sectors
    
    async def _screen_batch(self, fmp, symbols: List[str]) -> List[QualifiedCompany]:
        """
        Screen a batch of companies for business quality.
        
        Args:
            fmp: FMP client instance
            symbols: Batch of symbols to screen
            
        Returns:
            List of qualified companies from this batch
        """
        qualified = []
        
        for symbol in symbols:
            try:
                result = await self.screen_company(symbol, fmp)
                if result and result.quality_score.passed:
                    qualified.append(result)
                    
            except Exception as e:
                logger.debug(f"Error screening {symbol}: {e}")
                continue
        
        return qualified
    
    async def screen_company(self, symbol: str, fmp=None) -> Optional[QualifiedCompany]:
        """
        Screen individual company for business quality.
        
        Args:
            symbol: Stock symbol to screen
            fmp: Optional FMP client instance
            
        Returns:
            QualifiedCompany if passed, None if failed
        """
        try:
            # Get FMP client if not provided
            if fmp is None:
                async with await get_fmp_client(self.fmp_api_key) as fmp_client:
                    return await self._screen_company_impl(symbol, fmp_client)
            else:
                return await self._screen_company_impl(symbol, fmp)
                
        except Exception as e:
            logger.error(f"Error screening company {symbol}: {e}")
            return None
    
    async def _screen_company_impl(self, symbol: str, fmp) -> Optional[QualifiedCompany]:
        """Implementation of company screening"""
        try:
            # Get complete financial data
            financial_data = await fmp.get_complete_financial_data(symbol, years=10)
            if not financial_data or not financial_data.get('years_data'):
                logger.debug(f"{symbol}: No financial data available")
                return None
            
            profile = financial_data.get('profile')
            if not profile:
                logger.debug(f"{symbol}: No company profile available")
                return None
            
            # Apply the 5 filters
            filter_results = {}
            scores = {}
            disqualifications = []
            
            # Filter 1: Revenue Growth Consistency (20 points)
            revenue_result = self._filter_revenue_growth(financial_data)
            filter_results['revenue_growth'] = revenue_result
            scores['revenue_growth'] = revenue_result['score']
            if revenue_result['disqualified']:
                disqualifications.extend(revenue_result['reasons'])
            
            # Filter 2: Profitability & Cash Flow Strength (15 points)
            profit_result = self._filter_profitability(financial_data)
            filter_results['profitability'] = profit_result
            scores['profitability'] = profit_result['score']
            if profit_result['disqualified']:
                disqualifications.extend(profit_result['reasons'])
            
            # Filter 3: Balance Sheet Requirements (20 points)
            balance_result = self._filter_balance_sheet(financial_data)
            filter_results['balance_sheet'] = balance_result
            scores['balance_sheet'] = balance_result['score']
            if balance_result['disqualified']:
                disqualifications.extend(balance_result['reasons'])
            
            # Filter 4: Competitive Moat Metrics (25 points)
            moat_result = self._filter_competitive_moat(financial_data)
            filter_results['moat_strength'] = moat_result
            scores['moat_strength'] = moat_result['score']
            if moat_result['disqualified']:
                disqualifications.extend(moat_result['reasons'])
            
            # Filter 5: Management Quality Metrics (20 points)
            mgmt_result = self._filter_management_quality(financial_data)
            filter_results['management'] = mgmt_result
            scores['management'] = mgmt_result['score']
            if mgmt_result['disqualified']:
                disqualifications.extend(mgmt_result['reasons'])
            
            # Calculate total score
            total_score = sum(scores.values())
            max_score = (self.scoring_weights.revenue_growth_weight +
                        self.scoring_weights.profitability_weight +
                        self.scoring_weights.balance_sheet_weight +
                        self.scoring_weights.moat_strength_weight +
                        self.scoring_weights.management_quality_weight)
            
            passed = total_score >= self.scoring_weights.min_quality_score
            
            # Create quality score object
            quality_score = QualityScore(
                symbol=symbol,
                total_score=total_score,
                max_score=max_score,
                passed=passed,
                revenue_growth_score=scores['revenue_growth'],
                profitability_score=scores['profitability'],
                balance_sheet_score=scores['balance_sheet'],
                moat_strength_score=scores['moat_strength'],
                management_score=scores['management'],
                filter_results=filter_results,
                disqualification_reasons=disqualifications
            )
            
            if passed:
                # Create financial summary
                latest_year = max(financial_data['years_data'].keys())
                latest_data = financial_data['years_data'][latest_year]
                latest_ratios = financial_data['ratios'].get(latest_year)
                
                financial_summary = {
                    'revenue': latest_data['income'].revenue,
                    'net_income': latest_data['income'].net_income,
                    'free_cash_flow': latest_data['cashflow'].get('free_cash_flow', 0),
                    'roe': latest_ratios.roe if latest_ratios else 0,
                    'roic': latest_ratios.roic if latest_ratios else 0,
                    'debt_to_equity': latest_ratios.debt_to_equity if latest_ratios else 0,
                    'current_ratio': latest_ratios.current_ratio if latest_ratios else 0,
                    'gross_margin': latest_ratios.gross_margin if latest_ratios else 0
                }
                
                qualified_company = QualifiedCompany(
                    symbol=symbol,
                    company_name=profile.company_name,
                    sector=profile.sector,
                    industry=profile.industry,
                    market_cap=profile.market_cap,
                    quality_score=quality_score,
                    financial_summary=financial_summary
                )
                
                logger.info(f"{symbol} QUALIFIED: {total_score}/{max_score} points")
                return qualified_company
            else:
                logger.debug(f"{symbol} failed: {total_score}/{max_score} points - {disqualifications}")
                return None
                
        except Exception as e:
            logger.error(f"Error in company screening implementation for {symbol}: {e}")
            return None
    
    def _filter_revenue_growth(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Filter 1: Revenue Growth Consistency
        Must grow revenue in ≥8 of last 10 years with ≥5% CAGR
        """
        try:
            years_data = financial_data['years_data']
            sorted_years = sorted(years_data.keys(), reverse=True)[:10]  # Last 10 years
            
            if len(sorted_years) < 5:
                return {'score': 0, 'disqualified': True, 'reasons': ['Insufficient revenue history']}
            
            # Check year-over-year growth
            growth_years = 0
            revenues = []
            crisis_declines = {}
            
            for i, year in enumerate(sorted_years):
                revenue = years_data[year]['income'].revenue
                revenues.append((year, revenue))
                
                if i > 0:  # Compare to previous year
                    prev_year, prev_revenue = revenues[i-1]
                    if prev_revenue > 0:
                        growth = (revenue - prev_revenue) / prev_revenue
                        if growth > 0:
                            growth_years += 1
                        
                        # Check crisis years
                        if year in [2008, 2009]:
                            crisis_declines['2008_crisis'] = min(crisis_declines.get('2008_crisis', 0), growth)
                        elif year == 2020:
                            crisis_declines['2020_crisis'] = growth
            
            # Calculate CAGR
            if len(revenues) >= 5:
                latest_revenue = revenues[0][1]
                oldest_revenue = revenues[-1][1]
                years_span = len(revenues) - 1
                
                if oldest_revenue > 0:
                    cagr = ((latest_revenue / oldest_revenue) ** (1/years_span)) - 1
                else:
                    cagr = 0
            else:
                cagr = 0
            
            # Scoring
            score = 0
            reasons = []
            disqualified = False
            
            # Growth years (2 points each, max 20)
            score += min(growth_years * 2, self.scoring_weights.revenue_growth_weight)
            
            # Check minimum requirements
            if growth_years < self.criteria.min_revenue_growth_years:
                disqualified = True
                reasons.append(f"Revenue growth only in {growth_years}/{len(sorted_years)} years (need {self.criteria.min_revenue_growth_years})")
            
            if cagr < self.criteria.min_revenue_cagr:
                disqualified = True
                reasons.append(f"Revenue CAGR {cagr:.1%} below minimum {self.criteria.min_revenue_cagr:.1%}")
            
            # Check crisis resilience
            if '2008_crisis' in crisis_declines and crisis_declines['2008_crisis'] < -self.criteria.max_revenue_decline_2008:
                disqualified = True
                reasons.append(f"2008 crisis decline {crisis_declines['2008_crisis']:.1%} exceeded limit")
            
            if '2020_crisis' in crisis_declines and crisis_declines['2020_crisis'] < -self.criteria.max_revenue_decline_2020:
                disqualified = True
                reasons.append(f"2020 crisis decline {crisis_declines['2020_crisis']:.1%} exceeded limit")
            
            return {
                'score': score,
                'disqualified': disqualified,
                'reasons': reasons,
                'metrics': {
                    'growth_years': growth_years,
                    'total_years': len(sorted_years),
                    'revenue_cagr': cagr,
                    'crisis_declines': crisis_declines
                }
            }
            
        except Exception as e:
            logger.error(f"Error in revenue growth filter: {e}")
            return {'score': 0, 'disqualified': True, 'reasons': ['Revenue analysis failed']}
    
    def _filter_profitability(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Filter 2: Profitability & Cash Flow Strength
        Consistent profits and positive FCF
        """
        try:
            years_data = financial_data['years_data']
            sorted_years = sorted(years_data.keys(), reverse=True)[:10]
            
            profitable_years = 0
            fcf_positive_years = 0
            fcf_values = []
            
            for year in sorted_years:
                data = years_data[year]
                
                # Check profitability
                net_income = data['income'].net_income
                if net_income > 0:
                    profitable_years += 1
                
                # Check free cash flow
                fcf = data['cashflow'].get('free_cash_flow', 0)
                if fcf > 0:
                    fcf_positive_years += 1
                    fcf_values.append(fcf)
            
            # Calculate FCF growth
            fcf_cagr = 0
            if len(fcf_values) >= 5:
                try:
                    fcf_cagr = ((fcf_values[0] / fcf_values[-1]) ** (1/5)) - 1
                except:
                    fcf_cagr = 0
            
            # Scoring
            score = 0
            reasons = []
            disqualified = False
            
            # Profitable years scoring
            profit_score = min(profitable_years * 1.5, self.scoring_weights.profitability_weight * 0.6)
            score += profit_score
            
            # FCF positive years scoring  
            fcf_score = min(fcf_positive_years * 1.0, self.scoring_weights.profitability_weight * 0.4)
            score += fcf_score
            
            # Check minimum requirements
            if profitable_years < self.criteria.min_net_income_positive_years:
                disqualified = True
                reasons.append(f"Profitable only {profitable_years}/{len(sorted_years)} years (need {self.criteria.min_net_income_positive_years})")
            
            if fcf_positive_years < self.criteria.min_fcf_positive_years:
                disqualified = True
                reasons.append(f"Positive FCF only {fcf_positive_years}/{len(sorted_years)} years (need {self.criteria.min_fcf_positive_years})")
            
            if fcf_cagr < self.criteria.min_fcf_cagr:
                disqualified = True
                reasons.append(f"FCF CAGR {fcf_cagr:.1%} below minimum {self.criteria.min_fcf_cagr:.1%}")
            
            return {
                'score': int(score),
                'disqualified': disqualified,
                'reasons': reasons,
                'metrics': {
                    'profitable_years': profitable_years,
                    'fcf_positive_years': fcf_positive_years,
                    'fcf_cagr': fcf_cagr
                }
            }
            
        except Exception as e:
            logger.error(f"Error in profitability filter: {e}")
            return {'score': 0, 'disqualified': True, 'reasons': ['Profitability analysis failed']}
    
    def _filter_balance_sheet(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Filter 3: Balance Sheet Requirements
        Strong balance sheet with low debt and high liquidity
        """
        try:
            # Get latest year data
            latest_year = max(financial_data['years_data'].keys())
            latest_data = financial_data['years_data'][latest_year]
            latest_ratios = financial_data['ratios'].get(latest_year)
            
            if not latest_ratios:
                return {'score': 0, 'disqualified': True, 'reasons': ['No ratio data available']}
            
            # Extract metrics
            debt_to_equity = latest_ratios.debt_to_equity
            current_ratio = latest_ratios.current_ratio
            quick_ratio = latest_ratios.quick_ratio
            
            # Calculate additional metrics from balance sheet
            balance_data = latest_data['balance']
            cash = balance_data.get('cash_and_equivalents', 0)
            total_debt = balance_data.get('total_debt', 0)
            
            cash_to_debt_ratio = cash / total_debt if total_debt > 0 else float('inf')
            
            # Scoring (4 points per metric, 5 metrics = 20 points max)
            score = 0
            reasons = []
            disqualified = False
            
            # Debt-to-equity ratio
            if debt_to_equity <= self.criteria.max_debt_to_equity:
                score += 4
            else:
                disqualified = True
                reasons.append(f"Debt-to-equity {debt_to_equity:.2f} exceeds limit {self.criteria.max_debt_to_equity}")
            
            # Current ratio
            if current_ratio >= self.criteria.min_current_ratio:
                score += 4
            else:
                disqualified = True
                reasons.append(f"Current ratio {current_ratio:.2f} below minimum {self.criteria.min_current_ratio}")
            
            # Quick ratio
            if quick_ratio >= self.criteria.min_quick_ratio:
                score += 4
            else:
                disqualified = True
                reasons.append(f"Quick ratio {quick_ratio:.2f} below minimum {self.criteria.min_quick_ratio}")
            
            # Cash-to-debt ratio
            if cash_to_debt_ratio >= self.criteria.min_cash_to_debt_ratio:
                score += 4
            else:
                reasons.append(f"Cash-to-debt ratio {cash_to_debt_ratio:.2f} below preferred {self.criteria.min_cash_to_debt_ratio}")
                score += 2  # Partial credit
            
            # Interest coverage (approximate from data available)
            operating_income = latest_data['income'].operating_income
            if operating_income > 0 and total_debt > 0:
                # Estimate interest expense as 4% of total debt
                estimated_interest = total_debt * 0.04
                interest_coverage = operating_income / estimated_interest if estimated_interest > 0 else float('inf')
                
                if interest_coverage >= self.criteria.min_interest_coverage:
                    score += 4
                else:
                    reasons.append(f"Estimated interest coverage {interest_coverage:.1f}x below minimum {self.criteria.min_interest_coverage}x")
                    score += 1  # Partial credit
            else:
                score += 2  # Neutral if can't calculate
            
            return {
                'score': min(score, self.scoring_weights.balance_sheet_weight),
                'disqualified': disqualified,
                'reasons': reasons,
                'metrics': {
                    'debt_to_equity': debt_to_equity,
                    'current_ratio': current_ratio,
                    'quick_ratio': quick_ratio,
                    'cash_to_debt_ratio': cash_to_debt_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error in balance sheet filter: {e}")
            return {'score': 0, 'disqualified': True, 'reasons': ['Balance sheet analysis failed']}
    
    def _filter_competitive_moat(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Filter 4: Competitive Moat Metrics
        Consistent high ROIC and strong margins indicating pricing power
        """
        try:
            years_data = financial_data['years_data']
            ratios_data = financial_data['ratios']
            sorted_years = sorted(years_data.keys(), reverse=True)[:10]
            
            # Collect ROIC and margin data
            roic_values = []
            gross_margins = []
            
            for year in sorted_years:
                if year in ratios_data:
                    ratio = ratios_data[year]
                    if ratio.roic > 0:
                        roic_values.append(ratio.roic)
                    if ratio.gross_margin > 0:
                        gross_margins.append(ratio.gross_margin)
            
            # Count years with ROIC >= 15%
            high_roic_years = sum(1 for roic in roic_values if roic >= self.criteria.min_roic_threshold)
            
            # Current margins
            latest_year = max(sorted_years)
            latest_ratios = ratios_data.get(latest_year)
            current_gross_margin = latest_ratios.gross_margin if latest_ratios else 0
            
            # Scoring (25 points max)
            score = 0
            reasons = []
            disqualified = False
            
            # ROIC consistency (15 points max)
            roic_score = min(high_roic_years * 2, 15)
            score += roic_score
            
            if high_roic_years < self.criteria.min_roic_years:
                disqualified = True
                reasons.append(f"High ROIC only {high_roic_years}/{len(roic_values)} years (need {self.criteria.min_roic_years})")
            
            # Gross margin strength (10 points max)
            if current_gross_margin >= self.criteria.min_gross_margin:
                score += 10
            else:
                disqualified = True
                reasons.append(f"Gross margin {current_gross_margin:.1%} below minimum {self.criteria.min_gross_margin:.1%}")
            
            # Margin trend analysis
            if len(gross_margins) >= 5:
                recent_avg = statistics.mean(gross_margins[:3])
                older_avg = statistics.mean(gross_margins[-3:])
                margin_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                
                if margin_trend >= 0:  # Stable or improving
                    # Already included in base scoring
                    pass
                else:
                    reasons.append(f"Declining margin trend: {margin_trend:.1%}")
            
            return {
                'score': min(score, self.scoring_weights.moat_strength_weight),
                'disqualified': disqualified,
                'reasons': reasons,
                'metrics': {
                    'high_roic_years': high_roic_years,
                    'total_years_analyzed': len(roic_values),
                    'current_gross_margin': current_gross_margin,
                    'avg_roic': statistics.mean(roic_values) if roic_values else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in competitive moat filter: {e}")
            return {'score': 0, 'disqualified': True, 'reasons': ['Moat analysis failed']}
    
    def _filter_management_quality(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Filter 5: Management Quality Metrics
        Efficient management of assets and working capital
        """
        try:
            ratios_data = financial_data['ratios']
            sorted_years = sorted(ratios_data.keys(), reverse=True)[:5]  # Last 5 years
            
            if len(sorted_years) < 3:
                return {'score': 0, 'disqualified': True, 'reasons': ['Insufficient ratio data']}
            
            # Collect ROE and efficiency metrics
            roe_values = []
            asset_turnover_values = []
            dso_values = []
            
            for year in sorted_years:
                ratio = ratios_data[year]
                if ratio.roe > 0:
                    roe_values.append(ratio.roe)
                if ratio.asset_turnover > 0:
                    asset_turnover_values.append(ratio.asset_turnover)
                if ratio.days_sales_outstanding > 0:
                    dso_values.append(ratio.days_sales_outstanding)
            
            # Calculate averages
            avg_roe = statistics.mean(roe_values) if roe_values else 0
            avg_asset_turnover = statistics.mean(asset_turnover_values) if asset_turnover_values else 0
            avg_dso = statistics.mean(dso_values) if dso_values else 0
            
            # Current year metrics
            latest_year = max(sorted_years)
            latest_ratios = ratios_data[latest_year]
            current_roe = latest_ratios.roe
            
            # Scoring (20 points max)
            score = 0
            reasons = []
            disqualified = False
            
            # ROE consistency and level (10 points)
            if avg_roe >= self.criteria.min_roe_5yr_avg:
                score += 8
                
                # Check current ROE stability
                roe_tolerance = abs(current_roe - avg_roe) / avg_roe if avg_roe > 0 else 1
                if roe_tolerance <= self.criteria.roe_current_tolerance:
                    score += 2
                else:
                    reasons.append(f"Current ROE {current_roe:.1%} deviates {roe_tolerance:.1%} from 5-yr avg")
            else:
                disqualified = True
                reasons.append(f"5-year avg ROE {avg_roe:.1%} below minimum {self.criteria.min_roe_5yr_avg:.1%}")
            
            # Asset turnover efficiency (5 points)
            if avg_asset_turnover >= self.criteria.min_asset_turnover:
                score += 5
            else:
                reasons.append(f"Asset turnover {avg_asset_turnover:.2f} below minimum {self.criteria.min_asset_turnover}")
                score += 2  # Partial credit
            
            # Working capital management (5 points)
            if avg_dso > 0 and avg_dso <= self.criteria.max_days_sales_outstanding:
                score += 5
            else:
                if avg_dso > self.criteria.max_days_sales_outstanding:
                    reasons.append(f"Days sales outstanding {avg_dso:.0f} exceeds limit {self.criteria.max_days_sales_outstanding}")
                score += 1  # Minimal credit
            
            return {
                'score': min(score, self.scoring_weights.management_quality_weight),
                'disqualified': disqualified,
                'reasons': reasons,
                'metrics': {
                    'avg_roe_5yr': avg_roe,
                    'current_roe': current_roe,
                    'avg_asset_turnover': avg_asset_turnover,
                    'avg_days_sales_outstanding': avg_dso
                }
            }
            
        except Exception as e:
            logger.error(f"Error in management quality filter: {e}")
            return {'score': 0, 'disqualified': True, 'reasons': ['Management analysis failed']}
    
    def export_results(self, qualified_companies: List[QualifiedCompany], 
                      filename: str = None) -> str:
        """
        Export screening results to CSV file.
        
        Args:
            qualified_companies: List of qualified companies
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"quality_screen_results_{timestamp}.csv"
            
            # Prepare data for export
            export_data = []
            for company in qualified_companies:
                score = company.quality_score
                summary = company.financial_summary
                
                export_data.append({
                    'Symbol': company.symbol,
                    'Company Name': company.company_name,
                    'Sector': company.sector,
                    'Industry': company.industry,
                    'Market Cap (B)': company.market_cap / 1e9,
                    'Quality Score': f"{score.total_score}/{score.max_score}",
                    'Revenue Growth Score': score.revenue_growth_score,
                    'Profitability Score': score.profitability_score,
                    'Balance Sheet Score': score.balance_sheet_score,
                    'Moat Strength Score': score.moat_strength_score,
                    'Management Score': score.management_score,
                    'Revenue (B)': summary.get('revenue', 0) / 1e9,
                    'Net Income (M)': summary.get('net_income', 0) / 1e6,
                    'Free Cash Flow (M)': summary.get('free_cash_flow', 0) / 1e6,
                    'ROE (%)': f"{summary.get('roe', 0):.1%}",
                    'ROIC (%)': f"{summary.get('roic', 0):.1%}",
                    'Debt/Equity': f"{summary.get('debt_to_equity', 0):.2f}",
                    'Current Ratio': f"{summary.get('current_ratio', 0):.2f}",
                    'Gross Margin (%)': f"{summary.get('gross_margin', 0):.1%}"
                })
            
            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(qualified_companies)} qualified companies to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""
    
    def get_screening_summary(self, qualified_companies: List[QualifiedCompany]) -> Dict[str, Any]:
        """
        Generate summary statistics for screening results.
        
        Args:
            qualified_companies: List of qualified companies
            
        Returns:
            Dict with summary statistics
        """
        try:
            if not qualified_companies:
                return {'total_qualified': 0}
            
            # Score statistics
            scores = [company.quality_score.total_score for company in qualified_companies]
            
            # Sector breakdown
            sector_counts = {}
            for company in qualified_companies:
                sector = company.sector
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Market cap breakdown
            market_caps = [company.market_cap for company in qualified_companies]
            
            # Financial metrics averages
            avg_roe = statistics.mean([company.financial_summary.get('roe', 0) 
                                     for company in qualified_companies])
            avg_roic = statistics.mean([company.financial_summary.get('roic', 0) 
                                      for company in qualified_companies])
            avg_debt_equity = statistics.mean([company.financial_summary.get('debt_to_equity', 0) 
                                             for company in qualified_companies])
            
            summary = {
                'total_qualified': len(qualified_companies),
                'score_statistics': {
                    'average_score': statistics.mean(scores),
                    'median_score': statistics.median(scores),
                    'min_score': min(scores),
                    'max_score': max(scores)
                },
                'sector_breakdown': dict(sorted(sector_counts.items(), 
                                              key=lambda x: x[1], reverse=True)),
                'market_cap_statistics': {
                    'average_market_cap_b': statistics.mean(market_caps) / 1e9,
                    'median_market_cap_b': statistics.median(market_caps) / 1e9,
                    'min_market_cap_b': min(market_caps) / 1e9,
                    'max_market_cap_b': max(market_caps) / 1e9
                },
                'financial_averages': {
                    'avg_roe': avg_roe,
                    'avg_roic': avg_roic,
                    'avg_debt_to_equity': avg_debt_equity
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating screening summary: {e}")
            return {'total_qualified': 0, 'error': str(e)}


# Convenience functions for external use
async def screen_sp500_for_quality(fmp_api_key: str) -> List[QualifiedCompany]:
    """
    Screen S&P 500 universe for business quality.
    
    Args:
        fmp_api_key: Financial Modeling Prep API key
        
    Returns:
        List of companies that passed all quality filters
    """
    screener = BusinessQualityScreener(fmp_api_key)
    return await screener.screen_universe()


async def screen_custom_list(symbols: List[str], fmp_api_key: str) -> List[QualifiedCompany]:
    """
    Screen custom list of symbols for business quality.
    
    Args:
        symbols: List of stock symbols to screen
        fmp_api_key: Financial Modeling Prep API key
        
    Returns:
        List of companies that passed all quality filters
    """
    screener = BusinessQualityScreener(fmp_api_key)
    return await screener.screen_universe(symbols)


async def quick_quality_check(symbol: str, fmp_api_key: str) -> Optional[QualifiedCompany]:
    """
    Quick quality check for individual stock.
    
    Args:
        symbol: Stock symbol to check
        fmp_api_key: Financial Modeling Prep API key
        
    Returns:
        QualifiedCompany if passed, None if failed
    """
    screener = BusinessQualityScreener(fmp_api_key)
    return await screener.screen_company(symbol)