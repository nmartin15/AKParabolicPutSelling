"""
Holy Grail Options Strategy - Quantified Criteria Configuration
Based on Adam Khoo's "Holy Grail" cash-secured put selling methodology

All thresholds are derived from the transcript analysis and represent
the exact numerical criteria for trade qualification.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class PreferredSector(Enum):
    """Sectors that meet Adam's quality requirements"""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare" 
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    COMMUNICATION_SERVICES = "Communication Services"
    SELECT_FINANCIALS = "Select Financials"  # Asset managers, exchanges only


class ExcludedSector(Enum):
    """Sectors automatically disqualified"""
    COMMODITIES = "Commodities"
    AIRLINES = "Airlines"
    REAL_ESTATE = "Real Estate"
    SHIPPING = "Shipping"
    TRADITIONAL_RETAIL = "Traditional Retail"


@dataclass
class BusinessQualityCriteria:
    """
    The 5-filter business quality screening system.
    Must pass ALL criteria to qualify for trading.
    """
    
    # Revenue Growth Consistency (Filter 1)
    min_revenue_growth_years: int = 8  # Must grow revenue in ≥8 of last 10 years
    min_revenue_cagr: float = 0.05     # 5% minimum 10-year CAGR
    max_revenue_decline_2008: float = 0.15  # <15% decline in 2008-2009 crisis
    max_revenue_decline_2020: float = 0.20  # <20% decline in 2020 crisis
    revenue_volatility_threshold: float = 0.20  # Revenue std dev <20% of mean
    recovery_time_limit_years: int = 3  # Must recover to new highs within 3 years
    
    # Profitability & Cash Flow Strength (Filter 2)
    min_net_income_positive_years: int = 8  # Profitable in ≥8 of 10 years
    min_fcf_positive_years: int = 9         # Positive FCF in ≥9 of 10 years
    min_fcf_cagr: float = 0.03             # 3% minimum FCF growth over 5 years
    operating_margin_tolerance: float = 0.20  # Current margin within 20% of 5-yr avg
    min_cash_conversion_ratio: float = 0.90   # Operating CF ≥ 90% of Net Income
    min_fcf_yield: float = 0.03            # FCF/Market Cap ≥ 3%
    
    # Balance Sheet Requirements (Filter 3)
    max_debt_to_equity: float = 0.5        # <0.5 for non-financial companies
    min_current_ratio: float = 1.5         # ≥1.5 liquidity ratio
    min_quick_ratio: float = 1.0           # ≥1.0 acid test
    min_cash_to_debt_ratio: float = 0.25   # Cash ≥ 25% of total debt
    min_interest_coverage: float = 5.0      # EBIT/Interest ≥ 5x
    min_debt_service_coverage: float = 3.0  # Operating CF/Debt Service ≥ 3x
    
    # Competitive Moat Metrics (Filter 4)
    min_gross_margin: float = 0.40         # ≥40% gross margins
    min_roic_years: int = 7                # ROIC ≥15% in ≥7 of 10 years
    min_roic_threshold: float = 0.15       # 15% minimum ROIC
    max_customer_concentration: float = 0.10  # No customer >10% of revenue
    min_patent_life_years: int = 5         # ≥5 years remaining patent protection
    market_position_requirement: int = 3    # Must be top 3 player OR growing share
    
    # Management Quality Metrics (Filter 5)
    min_roe_5yr_avg: float = 0.15          # 5-year average ROE ≥15%
    roe_current_tolerance: float = 0.25     # Current ROE within 25% of average
    min_asset_turnover: float = 1.0        # ≥1.0 and stable/improving
    max_days_sales_outstanding: int = 45    # DSO ≤45 days
    min_revenue_per_capex_dollar: float = 3.0  # Revenue growth per $ capex ≥ $3


@dataclass
class SectorScoringCriteria:
    """
    Quantified scoring for preferred sectors.
    Must score ≥7/10 to qualify.
    """
    
    # Technology Sector Requirements
    software_recurring_revenue_min: float = 0.70    # SaaS: ≥70% recurring revenue
    software_gross_margin_min: float = 0.75         # SaaS: ≥75% gross margins
    semiconductor_rd_spend_min: float = 0.15        # Semis: R&D ≥15% of revenue
    semiconductor_design_wins_years: int = 2        # ≥2 years design win visibility
    cloud_revenue_growth_min: float = 0.20          # Cloud: ≥20% revenue growth
    
    # Healthcare Sector Requirements  
    pharma_pipeline_value_multiple: float = 2.0     # Pipeline value ≥2x current revenue
    pharma_patent_cliff_years: int = 5              # Patent cliff >5 years out
    medtech_recurring_revenue_min: float = 0.30     # Medical devices: ≥30% recurring
    
    # Financial Sector Requirements (Select Only)
    asset_manager_aum_growth_min: float = 0.10      # AUM growth ≥10% annually
    exchange_volume_growth_required: bool = True     # Must show volume growth


@dataclass
class ValuationCriteria:
    """
    DCF-based intrinsic value calculation parameters.
    Based on conservative assumptions per Adam's methodology.
    """
    
    # DCF Model Parameters
    max_revenue_growth_rate: float = 0.15           # Cap growth assumptions at 15%
    fade_to_gdp_growth_year: int = 6               # Start fading to GDP growth in year 6
    terminal_gdp_growth_rate: float = 0.025        # 2.5% perpetual growth
    corporate_tax_rate: float = 0.21               # 21% corporate tax rate
    maintenance_capex_pct: float = 0.03            # 3% of revenue for maintenance
    
    # Discount Rate Calculation
    min_discount_rate: float = 0.08                # 8% minimum discount rate
    max_discount_rate: float = 0.15                # 15% maximum discount rate
    equity_risk_premium: float = 0.06              # 6% equity risk premium
    
    # Terminal Value Constraints
    max_terminal_value_pct: float = 0.80           # Terminal <80% of total DCF value
    min_terminal_roic: float = 0.08                # Terminal ROIC ≥ Cost of Capital
    
    # Entry Price Requirements
    max_price_to_fair_value: float = 1.0          # Price ≤ Fair Value (required)
    preferred_margin_of_safety: float = 0.15      # Prefer 15% below fair value
    min_upside_potential: float = 0.20            # ≥20% upside to fair value


@dataclass
class TechnicalEntryCriteria:
    """
    "Perfect Storm" technical conditions for entry.
    ALL conditions must align simultaneously.
    """
    
    # Parabolic Drop Requirements
    min_single_day_drop: float = 0.05              # ≥5% single-day decline
    min_multi_day_drop: float = 0.10               # ≥10% decline over 2-3 days
    min_velocity_acceleration: float = 0.80        # Each day ≥80% of prior day's drop
    min_volume_spike: float = 1.5                  # Volume ≥150% of 20-day average
    min_gap_down: float = 0.02                     # Prefer ≥2% opening gap down
    max_intraday_recovery: float = 0.50            # <50% retracement of intraday low
    
    # Support Level Requirements
    support_tolerance_pct: float = 0.02            # ±2% tolerance for previous lows
    ma_200_tolerance_pct: float = 0.03             # ±3% tolerance for 200-day MA
    round_number_tolerance_pct: float = 0.05       # ±5% for round numbers ($50, $100)
    min_support_tests: int = 2                     # Support tested ≥2 times in 12 months
    min_support_volume_spike: float = 1.2          # Volume ≥120% when hitting support


@dataclass
class VolatilityEntryCriteria:
    """
    VIX and implied volatility requirements for optimal entry.
    High volatility = high option premiums.
    """
    
    # VIX Requirements
    min_vix_level: float = 30.0                    # VIX ≥30 (fear threshold)
    optimal_vix_min: float = 35.0                  # Optimal range starts at 35
    optimal_vix_max: float = 60.0                  # Optimal range ends at 60
    vix_percentile_threshold: float = 75.0         # ≥75th percentile of 252-day range
    min_vix_spike_multiple: float = 1.5            # VIX ≥1.5x its 20-day MA
    vix_entry_window_days: int = 5                 # Enter within 5 days of VIX spike
    
    # Individual Stock IV Requirements
    min_iv_percentile: float = 60.0                # ≥60th percentile of 252-day IV
    min_iv_vs_hv_ratio: float = 1.3               # IV ≥1.3x Historical Volatility
    min_put_skew_ratio: float = 1.05               # Put IV ≥105% of ATM call IV
    min_iv_expansion_ratio: float = 1.2            # Current IV ≥1.2x previous week


@dataclass
class OptionsSelectionCriteria:
    """
    Precise options chain analysis for optimal put selection.
    """
    
    # Strike Selection
    min_otm_distance: float = 0.08                 # Strike 8% below current price (min)
    max_otm_distance: float = 0.15                 # Strike 15% below current price (max)
    max_strike_to_fair_value: float = 1.05         # Strike ≤105% of DCF fair value
    min_delta: float = -0.35                       # Delta between -0.35 and -0.15
    max_delta: float = -0.15
    min_prob_otm: float = 0.70                     # ≥70% probability expire worthless
    
    # Premium Requirements
    min_premium_pct: float = 0.02                  # ≥2% of strike price
    min_premium_per_day: float = 0.0005            # ≥0.05% of strike per day
    min_risk_adjusted_return: float = 0.15         # Premium/(Strike-Current) ≥15%
    
    # Expiration and Liquidity
    min_dte: int = 30                              # 30-45 days to expiration
    max_dte: int = 45
    optimal_dte: int = 35                          # 35 days is optimal
    max_bid_ask_spread_dollars: float = 0.05       # ≤$0.05 or 5% of mid (whichever greater)
    max_bid_ask_spread_pct: float = 0.05
    min_open_interest: int = 100                   # ≥100 contracts open interest
    min_volume_5day: int = 50                      # ≥50 contracts traded in 5 days
    earnings_buffer_days: int = 0                  # No earnings during option period


@dataclass
class RiskManagementCriteria:
    """
    Position sizing and portfolio risk limits.
    Conservative approach to preserve capital.
    """
    
    # Position Sizing
    max_position_size_pct: float = 0.04            # 4% max per position
    max_sector_concentration_pct: float = 0.15     # 15% max per sector
    max_single_stock_exposure_pct: float = 0.08    # 8% max including potential assignment
    max_total_options_pct: float = 0.25            # 25% max options exposure
    min_cash_buffer_pct: float = 0.20              # 20% cash buffer above requirements
    
    # Portfolio Limits
    max_concurrent_positions: int = 10             # ≤10 active put positions
    max_correlated_positions: int = 3              # ≤3 positions with correlation >0.7
    correlation_threshold: float = 0.7             # Correlation limit
    target_portfolio_volatility: float = 0.15     # Target 15% annual portfolio volatility


@dataclass
class ExitCriteria:
    """
    Quantified rules for closing positions and taking profits.
    """
    
    # Time-Based Exits
    profit_target_pct: float = 0.50                # Close at 50% of max premium
    dte_evaluation_threshold: int = 21             # Evaluate closing at 21 DTE
    decay_threshold_pct: float = 0.80              # Close if option ≤20% of original premium
    min_daily_theta: float = 5.0                   # Close if daily theta <$5 per contract
    
    # Rolling Decision Matrix
    rolling_dte_threshold: int = 21                # Consider rolling at 21 DTE
    rolling_proximity_pct: float = 0.05            # Roll if stock within 5% of strike
    min_additional_premium: float = 0.50           # Need ≥$0.50 additional premium
    max_rolling_attempts: int = 3                  # Maximum 3 rolls per position
    min_rolling_premium: float = 0.25              # Don't roll for <$0.25 premium
    rolling_dte_extension: int = 35                # Add 30-45 days when rolling
    
    # Assignment Management
    max_assigned_cost_to_fv: float = 1.1          # Assigned cost ≤110% of fair value
    covered_call_strike_multiple: float = 1.1     # Sell calls at 110% of cost basis
    assignment_stop_loss_pct: float = 0.20        # 20% stop loss on assigned shares


@dataclass
class StopTradingCriteria:
    """
    Conditions that halt trading activity for risk management.
    """
    
    # Individual Stock Disqualification
    max_revenue_decline_yoy: float = 0.20          # >20% YoY revenue decline
    max_margin_compression_bps: int = 300          # >300 bps operating margin decline
    max_debt_increase: float = 0.2                 # Debt-to-equity increase >0.2
    min_roic_consecutive_qtrs: float = 0.10        # ROIC <10% for 2 consecutive quarters
    
    # Market Environment Triggers
    max_low_vix_days: int = 30                     # VIX <20 for >30 consecutive days
    min_avg_iv_percentile: float = 40.0            # Avg IV rank <40th percentile for 60 days
    min_avg_premium_yield: float = 0.015           # Avg premium yield <1.5% monthly
    min_strategy_hit_rate: float = 0.65            # Hit rate <65% over 20 trades
    min_strategy_sharpe_ratio: float = 1.0         # Sharpe ratio <1.0 over 6 months


@dataclass
class ScoringWeights:
    """
    Automated scoring system weights for trade qualification.
    """
    
    # Stock Quality Score (must be ≥80/100)
    revenue_growth_weight: int = 20                # 20 points max
    profitability_weight: int = 15                 # 15 points max
    balance_sheet_weight: int = 20                 # 20 points max
    moat_strength_weight: int = 25                 # 25 points max
    management_quality_weight: int = 20            # 20 points max
    min_quality_score: int = 80                    # Minimum passing score
    
    # Entry Signal Score (must be ≥90/100)
    price_action_weight: int = 30                  # 30 points max (15+15 for drop+volume)
    support_level_weight: int = 20                 # 20 points max
    volatility_weight: int = 25                    # 25 points max (15+10 for VIX+IV)
    options_metrics_weight: int = 25               # 25 points max (15+10 for premium+liquidity)
    min_entry_score: int = 90                      # Minimum passing score
    
    # Risk Assessment Score (must be ≤30/100)
    concentration_risk_weight: int = 10            # Penalty points
    correlation_risk_weight: int = 10              # Penalty points
    liquidity_risk_weight: int = 5                 # Penalty points
    fundamental_risk_weight: int = 5               # Penalty points
    max_risk_score: int = 30                       # Maximum acceptable risk


# Global configuration instance
CRITERIA = {
    'business_quality': BusinessQualityCriteria(),
    'sector_scoring': SectorScoringCriteria(),
    'valuation': ValuationCriteria(),
    'technical_entry': TechnicalEntryCriteria(),
    'volatility_entry': VolatilityEntryCriteria(),
    'options_selection': OptionsSelectionCriteria(),
    'risk_management': RiskManagementCriteria(),
    'exit_rules': ExitCriteria(),
    'stop_conditions': StopTradingCriteria(),
    'scoring_weights': ScoringWeights(),
    'preferred_sectors': [sector.value for sector in PreferredSector],
    'excluded_sectors': [sector.value for sector in ExcludedSector]
}


def get_criteria(category: str = None):
    """
    Get configuration criteria.
    
    Args:
        category: Specific category to retrieve, or None for all criteria
        
    Returns:
        Dict or specific criteria dataclass
    """
    if category is None:
        return CRITERIA
    return CRITERIA.get(category)


def validate_trade_opportunity(stock_data: dict, options_data: dict, market_data: dict) -> dict:
    """
    Validate if a trade opportunity meets all criteria.
    
    Args:
        stock_data: Fundamental and technical stock data
        options_data: Options chain and Greeks data  
        market_data: VIX, IV, and market condition data
        
    Returns:
        Dict with validation results and scores
    """
    # This will be implemented in the validation module
    pass