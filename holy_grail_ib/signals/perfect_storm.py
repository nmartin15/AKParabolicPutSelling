"""
Perfect Storm Detector for Holy Grail Options Strategy
Identifies Adam Khoo's "Holy Grail" entry conditions when ALL criteria align:

1. High-quality business (from screener)
2. Trading at/below intrinsic value (from DCF)
3. Parabolic drop with volume spike
4. VIX elevated (market fear)
5. Strong technical support level
6. High implied volatility (rich option premiums)

This is the core signal detection system that triggers actual trading opportunities.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statistics

from config.criteria import get_criteria
from data.ib_client import get_ib_client, StockData
from data.fmp_client import get_fmp_client
from screening.business_quality import BusinessQualityScreener, QualifiedCompany
from screening.valuation import DCFValuationEngine, DCFOutputs
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TechnicalSignal:
    """Technical analysis signal components"""
    symbol: str
    current_price: float
    
    # Price action signals
    single_day_drop_pct: float
    multi_day_drop_pct: float
    volume_spike_ratio: float
    gap_down_pct: float
    intraday_recovery_pct: float
    
    # Support level analysis
    support_level: float
    support_strength: int  # Number of times tested
    distance_to_support_pct: float
    support_type: str  # "previous_low", "ma_200", "round_number", etc.
    
    # Trend context
    ma_50: float
    ma_150: float
    ma_200: float
    trend_direction: str  # "uptrend", "downtrend", "sideways"
    
    # Validation flags
    parabolic_drop_confirmed: bool
    support_level_confirmed: bool
    volume_spike_confirmed: bool


@dataclass
class VolatilitySignal:
    """Volatility and options market signals"""
    symbol: str
    
    # VIX and market volatility
    current_vix: float
    vix_percentile: float
    vix_spike_confirmed: bool
    
    # Individual stock IV
    implied_volatility: float
    iv_percentile: float
    iv_vs_hv_ratio: float
    put_skew_ratio: float
    iv_expansion_confirmed: bool
    
    # Options market conditions
    options_volume: int
    put_call_ratio: float
    options_liquidity_score: float


@dataclass
class PerfectStormSignal:
    """Complete perfect storm signal when all conditions align"""
    symbol: str
    company_name: str
    sector: str
    timestamp: datetime
    
    # Signal strength (0-100)
    signal_strength: int
    entry_score: int  # Must be ≥90 to qualify
    
    # Component scores
    quality_score: int
    valuation_score: int
    technical_score: int
    volatility_score: int
    
    # Key metrics
    current_price: float
    intrinsic_value: float
    margin_of_safety: float
    support_level: float
    current_vix: float
    
    # Supporting data
    quality_data: QualifiedCompany
    valuation_data: DCFOutputs
    technical_data: TechnicalSignal
    volatility_data: VolatilitySignal
    
    # Trading readiness
    ready_for_options_analysis: bool
    estimated_put_strike_range: Tuple[float, float]
    expected_premium_yield: float
    
    # Alerts and warnings
    signal_warnings: List[str]
    urgency_level: str  # "low", "medium", "high", "critical"


class PerfectStormDetector:
    """
    Detects Adam's "Perfect Storm" conditions for cash-secured put selling.
    
    Combines business quality, valuation, technical analysis, and volatility
    to identify optimal entry opportunities.
    """
    
    def __init__(self, fmp_api_key: str, ib_host: str = '127.0.0.1', ib_port: int = 7497):
        """
        Initialize Perfect Storm Detector.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key
            ib_host: Interactive Brokers host
            ib_port: Interactive Brokers port
        """
        self.fmp_api_key = fmp_api_key
        self.ib_host = ib_host
        self.ib_port = ib_port
        
        # Load criteria
        self.technical_criteria = get_criteria('technical_entry')
        self.volatility_criteria = get_criteria('volatility_entry')
        self.scoring_weights = get_criteria('scoring_weights')
        
        # Component engines
        self.quality_screener = BusinessQualityScreener(fmp_api_key)
        self.valuation_engine = DCFValuationEngine(fmp_api_key)
        
        # Watchlist of qualified companies
        self.qualified_watchlist: Dict[str, QualifiedCompany] = {}
        self.intrinsic_values: Dict[str, DCFOutputs] = {}
        
        # Signal history
        self.signal_history: List[PerfectStormSignal] = []
        self.last_scan_time: Optional[datetime] = None
    
    async def initialize_watchlist(self, custom_symbols: List[str] = None) -> int:
        """
        Initialize watchlist with quality-screened companies and their valuations.
        
        Args:
            custom_symbols: Optional custom symbol list, otherwise uses S&P 500
            
        Returns:
            Number of qualified companies in watchlist
        """
        try:
            logger.info("Initializing perfect storm watchlist...")
            
            # Step 1: Screen for business quality
            if custom_symbols:
                qualified_companies = await self.quality_screener.screen_universe(custom_symbols)
            else:
                qualified_companies = await self.quality_screener.screen_universe()
            
            logger.info(f"Business quality screening: {len(qualified_companies)} companies qualified")
            
            # Step 2: Calculate intrinsic values
            symbols = [company.symbol for company in qualified_companies]
            valuations = await self.valuation_engine.batch_valuations(symbols)
            
            logger.info(f"DCF valuations completed: {len(valuations)} companies valued")
            
            # Step 3: Build watchlist (quality + undervalued)
            self.qualified_watchlist = {}
            self.intrinsic_values = {}
            
            for company in qualified_companies:
                symbol = company.symbol
                if symbol in valuations:
                    valuation = valuations[symbol]
                    
                    # Only include if reasonably valued
                    if valuation.passes_valuation_criteria:
                        self.qualified_watchlist[symbol] = company
                        self.intrinsic_values[symbol] = valuation
            
            logger.info(f"Perfect storm watchlist initialized: {len(self.qualified_watchlist)} companies ready")
            return len(self.qualified_watchlist)
            
        except Exception as e:
            logger.error(f"Error initializing watchlist: {e}")
            return 0
    
    async def scan_for_perfect_storms(self, force_refresh: bool = False) -> List[PerfectStormSignal]:
        """
        Scan entire watchlist for perfect storm conditions.
        
        Args:
            force_refresh: Force refresh of technical data
            
        Returns:
            List of perfect storm signals detected
        """
        try:
            if not self.qualified_watchlist:
                logger.warning("Watchlist not initialized. Call initialize_watchlist() first.")
                return []
            
            logger.info(f"Scanning {len(self.qualified_watchlist)} companies for perfect storm conditions...")
            
            # Get IB client for real-time data
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            
            # Get current VIX level
            current_vix = await ib_client.get_vix_data()
            if not current_vix:
                logger.warning("Could not retrieve VIX data")
                return []
            
            # Check if market conditions are favorable for perfect storms
            if not self._is_market_favorable_for_signals(current_vix):
                logger.info(f"Market conditions not favorable: VIX {current_vix:.1f}")
                return []
            
            # Scan each company in watchlist
            perfect_storms = []
            symbols = list(self.qualified_watchlist.keys())
            
            # Process in batches to manage API limits
            batch_size = 10
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_storms = await self._scan_batch(ib_client, batch, current_vix)
                perfect_storms.extend(batch_storms)
                
                logger.info(f"Batch {i//batch_size + 1}: {len(batch_storms)} signals detected")
                await asyncio.sleep(1)  # Rate limiting
            
            # Sort by signal strength
            perfect_storms.sort(key=lambda x: x.signal_strength, reverse=True)
            
            # Update signal history
            self.signal_history.extend(perfect_storms)
            self.last_scan_time = datetime.now()
            
            logger.info(f"Perfect storm scan complete: {len(perfect_storms)} signals detected")
            
            return perfect_storms
            
        except Exception as e:
            logger.error(f"Error scanning for perfect storms: {e}")
            return []
    
    async def _scan_batch(self, ib_client, symbols: List[str], current_vix: float) -> List[PerfectStormSignal]:
        """Scan batch of symbols for perfect storm conditions"""
        batch_storms = []
        
        for symbol in symbols:
            try:
                storm_signal = await self._analyze_symbol_for_perfect_storm(
                    ib_client, symbol, current_vix
                )
                
                if storm_signal and storm_signal.entry_score >= self.scoring_weights.min_entry_score:
                    batch_storms.append(storm_signal)
                    logger.info(f"PERFECT STORM DETECTED: {symbol} - Score: {storm_signal.entry_score}/100")
                
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue
        
        return batch_storms
    
    async def _analyze_symbol_for_perfect_storm(self, ib_client, symbol: str, 
                                              current_vix: float) -> Optional[PerfectStormSignal]:
        """
        Analyze individual symbol for perfect storm conditions.
        
        Args:
            ib_client: IB client instance
            symbol: Stock symbol to analyze
            current_vix: Current VIX level
            
        Returns:
            PerfectStormSignal if conditions met, None otherwise
        """
        try:
            # Get company data
            company = self.qualified_watchlist[symbol]
            valuation = self.intrinsic_values[symbol]
            
            # Get real-time stock data
            stock_data = await ib_client.get_stock_data(symbol)
            if not stock_data:
                return None
            
            # Update current price in valuation
            valuation.current_price = stock_data.price
            valuation.margin_of_safety = (valuation.intrinsic_value_per_share - stock_data.price) / stock_data.price
            valuation.upside_potential = (valuation.intrinsic_value_per_share / stock_data.price) - 1
            
            # Check if price is at/below intrinsic value (fundamental requirement)
            if stock_data.price > valuation.intrinsic_value_per_share:
                return None  # Not undervalued enough
            
            # Analyze technical signals
            technical_signal = await self._analyze_technical_signals(ib_client, symbol, stock_data)
            if not technical_signal.parabolic_drop_confirmed:
                return None  # No parabolic drop
            
            # Analyze volatility signals
            volatility_signal = await self._analyze_volatility_signals(ib_client, symbol, current_vix)
            if not volatility_signal.vix_spike_confirmed:
                return None  # VIX not elevated enough
            
            # Calculate component scores
            quality_score = company.quality_score.total_score
            valuation_score = self._score_valuation_attractiveness(valuation)
            technical_score = self._score_technical_signals(technical_signal)
            volatility_score = self._score_volatility_signals(volatility_signal)
            
            # Calculate overall entry score
            entry_score = min(technical_score + volatility_score, 100)
            signal_strength = (quality_score + valuation_score + technical_score + volatility_score) // 4
            
            # Determine urgency level
            urgency_level = self._determine_urgency(technical_signal, volatility_signal)
            
            # Estimate options parameters
            strike_range, expected_yield = self._estimate_options_parameters(
                stock_data.price, valuation.intrinsic_value_per_share, volatility_signal.implied_volatility
            )
            
            # Check final qualification
            ready_for_options = (entry_score >= self.scoring_weights.min_entry_score and
                               valuation_score >= 70 and  # Attractive valuation
                               quality_score >= 80)       # High quality (from criteria)
            
            # Create perfect storm signal
            perfect_storm = PerfectStormSignal(
                symbol=symbol,
                company_name=company.company_name,
                sector=company.sector,
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                entry_score=entry_score,
                quality_score=quality_score,
                valuation_score=valuation_score,
                technical_score=technical_score,
                volatility_score=volatility_score,
                current_price=stock_data.price,
                intrinsic_value=valuation.intrinsic_value_per_share,
                margin_of_safety=valuation.margin_of_safety,
                support_level=technical_signal.support_level,
                current_vix=current_vix,
                quality_data=company,
                valuation_data=valuation,
                technical_data=technical_signal,
                volatility_data=volatility_signal,
                ready_for_options_analysis=ready_for_options,
                estimated_put_strike_range=strike_range,
                expected_premium_yield=expected_yield,
                signal_warnings=[],
                urgency_level=urgency_level
            )
            
            return perfect_storm
            
        except Exception as e:
            logger.error(f"Error analyzing perfect storm for {symbol}: {e}")
            return None
    
    def _is_market_favorable_for_signals(self, vix: float) -> bool:
        """Check if overall market conditions favor perfect storm signals"""
        return vix >= self.volatility_criteria.min_vix_level
    
    async def _analyze_technical_signals(self, ib_client, symbol: str, 
                                       stock_data: StockData) -> TechnicalSignal:
        """
        Analyze technical price action for parabolic drop and support levels.
        
        Args:
            ib_client: IB client instance
            symbol: Stock symbol
            stock_data: Current stock data
            
        Returns:
            TechnicalSignal with analysis results
        """
        try:
            # Get historical data for analysis
            hist_data = await ib_client.get_historical_data(symbol, '2 M', '1 day')
            if hist_data is None or len(hist_data) < 50:
                # Return minimal signal if no historical data
                return TechnicalSignal(
                    symbol=symbol,
                    current_price=stock_data.price,
                    single_day_drop_pct=abs(stock_data.change_pct),
                    multi_day_drop_pct=0,
                    volume_spike_ratio=stock_data.volume / max(stock_data.avg_volume_20d, 1),
                    gap_down_pct=0,
                    intraday_recovery_pct=0,
                    support_level=stock_data.price * 0.95,  # Estimate
                    support_strength=1,
                    distance_to_support_pct=0.05,
                    support_type="estimated",
                    ma_50=stock_data.price,
                    ma_150=stock_data.price,
                    ma_200=stock_data.price,
                    trend_direction="unknown",
                    parabolic_drop_confirmed=abs(stock_data.change_pct) >= self.technical_criteria.min_single_day_drop * 100,
                    support_level_confirmed=False,
                    volume_spike_confirmed=stock_data.volume >= stock_data.avg_volume_20d * self.technical_criteria.min_volume_spike
                )
            
            # Calculate moving averages
            hist_data['ma_50'] = hist_data['close'].rolling(50).mean()
            hist_data['ma_150'] = hist_data['close'].rolling(150).mean()
            hist_data['ma_200'] = hist_data['close'].rolling(200).mean()
            
            current_ma_50 = hist_data['ma_50'].iloc[-1]
            current_ma_150 = hist_data['ma_150'].iloc[-1]
            current_ma_200 = hist_data['ma_200'].iloc[-1]
            
            # Determine trend direction
            if current_ma_50 > current_ma_150 > current_ma_200:
                trend_direction = "uptrend"
            elif current_ma_50 < current_ma_150 < current_ma_200:
                trend_direction = "downtrend"
            else:
                trend_direction = "sideways"
            
            # Analyze price drops
            recent_prices = hist_data['close'].tail(5).tolist()
            recent_volumes = hist_data['volume'].tail(5).tolist()
            
            # Single day drop
            single_day_drop = abs(stock_data.change_pct / 100)
            
            # Multi-day drop (last 3 days)
            if len(recent_prices) >= 3:
                multi_day_drop = (recent_prices[-3] - recent_prices[-1]) / recent_prices[-3]
                multi_day_drop = max(multi_day_drop, 0)
            else:
                multi_day_drop = single_day_drop
            
            # Volume analysis
            avg_volume = hist_data['volume'].tail(20).mean()
            volume_spike_ratio = stock_data.volume / avg_volume if avg_volume > 0 else 1
            
            # Gap analysis (simplified)
            if len(hist_data) >= 2:
                prev_close = hist_data['close'].iloc[-2]
                today_open = hist_data['open'].iloc[-1] if 'open' in hist_data.columns else stock_data.price
                gap_down_pct = max(0, (prev_close - today_open) / prev_close)
            else:
                gap_down_pct = 0
            
            # Intraday recovery (simplified)
            if stock_data.high > stock_data.low:
                intraday_recovery_pct = (stock_data.price - stock_data.low) / (stock_data.high - stock_data.low)
            else:
                intraday_recovery_pct = 0
            
            # Support level analysis
            support_level, support_strength, support_type = self._find_support_level(hist_data, stock_data.price)
            distance_to_support = abs(stock_data.price - support_level) / stock_data.price
            
            # Validate signals
            parabolic_drop_confirmed = (
                single_day_drop >= self.technical_criteria.min_single_day_drop or
                multi_day_drop >= self.technical_criteria.min_multi_day_drop
            )
            
            volume_spike_confirmed = volume_spike_ratio >= self.technical_criteria.min_volume_spike
            support_level_confirmed = distance_to_support <= 0.05  # Within 5% of support
            
            return TechnicalSignal(
                symbol=symbol,
                current_price=stock_data.price,
                single_day_drop_pct=single_day_drop * 100,
                multi_day_drop_pct=multi_day_drop * 100,
                volume_spike_ratio=volume_spike_ratio,
                gap_down_pct=gap_down_pct * 100,
                intraday_recovery_pct=intraday_recovery_pct * 100,
                support_level=support_level,
                support_strength=support_strength,
                distance_to_support_pct=distance_to_support * 100,
                support_type=support_type,
                ma_50=current_ma_50,
                ma_150=current_ma_150,
                ma_200=current_ma_200,
                trend_direction=trend_direction,
                parabolic_drop_confirmed=parabolic_drop_confirmed,
                support_level_confirmed=support_level_confirmed,
                volume_spike_confirmed=volume_spike_confirmed
            )
            
        except Exception as e:
            logger.error(f"Error analyzing technical signals for {symbol}: {e}")
            # Return default signal
            return TechnicalSignal(
                symbol=symbol,
                current_price=stock_data.price,
                single_day_drop_pct=0,
                multi_day_drop_pct=0,
                volume_spike_ratio=1,
                gap_down_pct=0,
                intraday_recovery_pct=50,
                support_level=stock_data.price * 0.95,
                support_strength=1,
                distance_to_support_pct=5,
                support_type="unknown",
                ma_50=stock_data.price,
                ma_150=stock_data.price,
                ma_200=stock_data.price,
                trend_direction="unknown",
                parabolic_drop_confirmed=False,
                support_level_confirmed=False,
                volume_spike_confirmed=False
            )
    
    def _find_support_level(self, hist_data: pd.DataFrame, current_price: float) -> Tuple[float, int, str]:
        """
        Find nearest significant support level.
        
        Args:
            hist_data: Historical price data
            current_price: Current stock price
            
        Returns:
            Tuple of (support_level, strength, type)
        """
        try:
            # Look for previous lows (support)
            lows = hist_data['low'].tail(252)  # Last year of data
            
            # Find significant lows (bottom 10% of range)
            low_threshold = lows.quantile(0.10)
            significant_lows = lows[lows <= low_threshold]
            
            if len(significant_lows) > 0:
                # Find closest support below current price
                support_candidates = significant_lows[significant_lows <= current_price * 1.02]
                
                if len(support_candidates) > 0:
                    support_level = support_candidates.max()  # Highest support below current
                    
                    # Count how many times this level was tested
                    tolerance = support_level * 0.02  # 2% tolerance
                    tests = len(lows[(lows >= support_level - tolerance) & 
                                   (lows <= support_level + tolerance)])
                    
                    return support_level, tests, "previous_low"
            
            # Fallback to moving average support
            if 'ma_200' in hist_data.columns and not hist_data['ma_200'].empty:
                ma_200 = hist_data['ma_200'].iloc[-1]
                if ma_200 <= current_price * 1.03:  # Within 3%
                    return ma_200, 1, "ma_200"
            
            # Fallback to round number support
            round_numbers = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
            for level in round_numbers:
                if level <= current_price * 1.05 and level >= current_price * 0.85:
                    return level, 1, "round_number"
            
            # Final fallback
            return current_price * 0.95, 1, "estimated"
            
        except Exception as e:
            logger.debug(f"Error finding support level: {e}")
            return current_price * 0.95, 1, "error"
    
    async def _analyze_volatility_signals(self, ib_client, symbol: str, 
                                        current_vix: float) -> VolatilitySignal:
        """
        Analyze volatility conditions for optimal option premiums.
        
        Args:
            ib_client: IB client instance
            symbol: Stock symbol
            current_vix: Current VIX level
            
        Returns:
            VolatilitySignal with analysis results
        """
        try:
            # VIX analysis (simplified - would need historical VIX data for percentile)
            vix_percentile = 75.0  # Assume elevated if VIX > 30
            vix_spike_confirmed = current_vix >= self.volatility_criteria.min_vix_level
            
            # Individual stock IV analysis (simplified - would need options data)
            # For now, estimate based on VIX relationship
            estimated_iv = current_vix * 0.8  # Stocks typically 80% of VIX
            iv_percentile = 70.0  # Assume elevated during market stress
            iv_vs_hv_ratio = 1.4   # Assume IV > HV during stress
            put_skew_ratio = 1.1    # Assume put premium during fear
            
            iv_expansion_confirmed = (
                estimated_iv > 25 and  # Minimum IV threshold
                iv_vs_hv_ratio >= self.volatility_criteria.min_iv_vs_hv_ratio
            )
            
            return VolatilitySignal(
                symbol=symbol,
                current_vix=current_vix,
                vix_percentile=vix_percentile,
                vix_spike_confirmed=vix_spike_confirmed,
                implied_volatility=estimated_iv,
                iv_percentile=iv_percentile,
                iv_vs_hv_ratio=iv_vs_hv_ratio,
                put_skew_ratio=put_skew_ratio,
                iv_expansion_confirmed=iv_expansion_confirmed,
                options_volume=1000,  # Placeholder
                put_call_ratio=1.2,   # Placeholder
                options_liquidity_score=0.8  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Error analyzing volatility signals for {symbol}: {e}")
            return VolatilitySignal(
                symbol=symbol,
                current_vix=current_vix,
                vix_percentile=50,
                vix_spike_confirmed=False,
                implied_volatility=20,
                iv_percentile=50,
                iv_vs_hv_ratio=1.0,
                put_skew_ratio=1.0,
                iv_expansion_confirmed=False,
                options_volume=0,
                put_call_ratio=1.0,
                options_liquidity_score=0.5
            )
    
    def _score_valuation_attractiveness(self, valuation: DCFOutputs) -> int:
        """Score valuation attractiveness (0-100)"""
        score = 50  # Base score
        
        # Margin of safety bonus
        if valuation.margin_of_safety > 0.20:  # >20% margin
            score += 30
        elif valuation.margin_of_safety > 0.10:  # >10% margin
            score += 20
        elif valuation.margin_of_safety > 0:     # Any margin
            score += 10
        
        # Upside potential bonus
        if valuation.upside_potential > 0.50:   # >50% upside
            score += 20
        elif valuation.upside_potential > 0.25: # >25% upside
            score += 10
        
        return min(score, 100)
    
    def _score_technical_signals(self, technical: TechnicalSignal) -> int:
        """Score technical signals (0-30 points from criteria)"""
        score = 0
        
        # Price action (15 points max)
        if technical.parabolic_drop_confirmed:
            drop_magnitude = max(technical.single_day_drop_pct, technical.multi_day_drop_pct)
            if drop_magnitude >= 10:      # ≥10% drop
                score += 15
            elif drop_magnitude >= 7:     # ≥7% drop
                score += 12
            elif drop_magnitude >= 5:     # ≥5% drop
                score += 10
        
        # Volume confirmation (15 points max)  
        if technical.volume_spike_confirmed:
            if technical.volume_spike_ratio >= 2.0:   # 2x+ volume
                score += 15
            elif technical.volume_spike_ratio >= 1.5: # 1.5x+ volume
                score += 12
            elif technical.volume_spike_ratio >= 1.2: # 1.2x+ volume
                score += 8
        
        return min(score, 30)
    
    def _score_volatility_signals(self, volatility: VolatilitySignal) -> int:
        """Score volatility signals (0-25 points from criteria)"""
        score = 0
        
        # VIX level (15 points max)
        if volatility.vix_spike_confirmed:
            if volatility.current_vix >= 50:      # Extreme fear
                score += 15
            elif volatility.current_vix >= 40:    # High fear
                score += 12
            elif volatility.current_vix >= 30:    # Moderate fear
                score += 10
        
        # IV expansion (10 points max)
        if volatility.iv_expansion_confirmed:
            if volatility.iv_vs_hv_ratio >= 1.5:  # Strong expansion
                score += 10
            elif volatility.iv_vs_hv_ratio >= 1.3: # Moderate expansion
                score += 7
            elif volatility.iv_vs_hv_ratio >= 1.1: # Slight expansion
                score += 5
        
        return min(score, 25)
    
    def _determine_urgency(self, technical: TechnicalSignal, 
                          volatility: VolatilitySignal) -> str:
        """Determine signal urgency level"""
        if (technical.single_day_drop_pct >= 10 and 
            volatility.current_vix >= 50):
            return "critical"
        elif (technical.single_day_drop_pct >= 7 and
              volatility.current_vix >= 40):
            return "high"
        elif (technical.single_day_drop_pct >= 5 and
              volatility.current_vix >= 35):
            return "medium"
        else:
            return "low"
    
    def _estimate_options_parameters(self, current_price: float, intrinsic_value: float, 
                                   implied_vol: float) -> Tuple[Tuple[float, float], float]:
        """
        Estimate optimal put strike range and expected premium yield.
        
        Args:
            current_price: Current stock price
            intrinsic_value: DCF intrinsic value
            implied_vol: Estimated implied volatility
            
        Returns:
            Tuple of ((strike_low, strike_high), expected_yield)
        """
        try:
            # Strike range: 8-15% below current price, but not above intrinsic value
            strike_high = min(current_price * 0.92, intrinsic_value * 1.05)  # 8% below or 105% of IV
            strike_low = current_price * 0.85   # 15% below
            
            # Ensure strikes make sense
            strike_high = max(strike_high, strike_low)
            
            # Estimate premium yield (simplified Black-Scholes approximation)
            # This is a rough estimate - real implementation would use proper options pricing
            time_to_expiry = 35 / 365  # 35 days in years
            moneyness = strike_high / current_price  # How far OTM
            
            # Rough premium estimate: IV * sqrt(time) * moneyness adjustment
            estimated_premium_pct = implied_vol / 100 * np.sqrt(time_to_expiry) * (1 - moneyness) * 2
            estimated_premium_pct = max(0.01, min(estimated_premium_pct, 0.10))  # 1-10% range
            
            # Annualized yield
            expected_yield = estimated_premium_pct * (365 / 35)  # Annualized
            
            return (strike_low, strike_high), expected_yield
            
        except Exception as e:
            logger.debug(f"Error estimating options parameters: {e}")
            return (current_price * 0.85, current_price * 0.92), 0.20  # Default estimates
    
    async def monitor_active_signals(self, signals: List[PerfectStormSignal]) -> List[PerfectStormSignal]:
        """
        Monitor previously detected signals for continued validity.
        
        Args:
            signals: List of active perfect storm signals
            
        Returns:
            List of signals that remain valid
        """
        try:
            if not signals:
                return []
            
            logger.info(f"Monitoring {len(signals)} active perfect storm signals...")
            
            # Get IB client
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            current_vix = await ib_client.get_vix_data()
            
            if not current_vix:
                logger.warning("Cannot monitor signals - VIX data unavailable")
                return signals  # Return unchanged if can't verify
            
            valid_signals = []
            
            for signal in signals:
                try:
                    # Check if signal is still fresh (within last 2 hours)
                    signal_age = datetime.now() - signal.timestamp
                    if signal_age > timedelta(hours=2):
                        logger.info(f"{signal.symbol}: Signal expired (age: {signal_age})")
                        continue
                    
                    # Get updated stock data
                    stock_data = await ib_client.get_stock_data(signal.symbol)
                    if not stock_data:
                        continue
                    
                    # Check if price is still near support and below intrinsic value
                    price_vs_support = abs(stock_data.price - signal.support_level) / signal.support_level
                    price_vs_intrinsic = stock_data.price / signal.intrinsic_value
                    
                    # Signal remains valid if:
                    # 1. Price still within 10% of support level
                    # 2. Price still at/below 105% of intrinsic value  
                    # 3. VIX still elevated
                    if (price_vs_support <= 0.10 and 
                        price_vs_intrinsic <= 1.05 and
                        current_vix >= self.volatility_criteria.min_vix_level):
                        
                        # Update signal with current data
                        signal.current_price = stock_data.price
                        signal.current_vix = current_vix
                        signal.margin_of_safety = (signal.intrinsic_value - stock_data.price) / stock_data.price
                        
                        valid_signals.append(signal)
                        logger.debug(f"{signal.symbol}: Signal remains valid")
                    else:
                        logger.info(f"{signal.symbol}: Signal invalidated - conditions changed")
                
                except Exception as e:
                    logger.debug(f"Error monitoring signal for {signal.symbol}: {e}")
                    continue
            
            logger.info(f"Signal monitoring complete: {len(valid_signals)}/{len(signals)} remain valid")
            return valid_signals
            
        except Exception as e:
            logger.error(f"Error monitoring active signals: {e}")
            return signals  # Return original list if monitoring fails
    
    def get_top_signals(self, signals: List[PerfectStormSignal], limit: int = 5) -> List[PerfectStormSignal]:
        """
        Get top perfect storm signals sorted by attractiveness.
        
        Args:
            signals: List of perfect storm signals
            limit: Maximum number of signals to return
            
        Returns:
            Top signals sorted by combined score
        """
        if not signals:
            return []
        
        # Sort by signal strength, then by entry score
        sorted_signals = sorted(signals, 
                               key=lambda x: (x.signal_strength, x.entry_score), 
                               reverse=True)
        
        return sorted_signals[:limit]
    
    def export_signals_report(self, signals: List[PerfectStormSignal], 
                             filename: str = None) -> str:
        """
        Export perfect storm signals to detailed report.
        
        Args:
            signals: List of perfect storm signals
            filename: Output filename
            
        Returns:
            Path to exported report
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"perfect_storm_signals_{timestamp}.csv"
            
            # Prepare export data
            export_data = []
            for signal in signals:
                export_data.append({
                    'Symbol': signal.symbol,
                    'Company': signal.company_name,
                    'Sector': signal.sector,
                    'Signal Strength': signal.signal_strength,
                    'Entry Score': f"{signal.entry_score}/100",
                    'Current Price': f"${signal.current_price:.2f}",
                    'Intrinsic Value': f"${signal.intrinsic_value:.2f}",
                    'Margin of Safety': f"{signal.margin_of_safety:.1%}",
                    'Support Level': f"${signal.support_level:.2f}",
                    'Distance to Support': f"{((signal.current_price - signal.support_level) / signal.support_level):.1%}",
                    'VIX Level': f"{signal.current_vix:.1f}",
                    'Drop %': f"{signal.technical_data.single_day_drop_pct:.1f}%",
                    'Volume Spike': f"{signal.technical_data.volume_spike_ratio:.1f}x",
                    'Est Strike Range': f"${signal.estimated_put_strike_range[0]:.0f}-${signal.estimated_put_strike_range[1]:.0f}",
                    'Est Premium Yield': f"{signal.expected_premium_yield:.1%}",
                    'Urgency': signal.urgency_level,
                    'Quality Score': f"{signal.quality_score}/100",
                    'Ready for Trading': signal.ready_for_options_analysis,
                    'Timestamp': signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Export to CSV
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(signals)} perfect storm signals to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting signals report: {e}")
            return ""
    
    def get_signal_summary(self, signals: List[PerfectStormSignal]) -> Dict[str, Any]:
        """
        Generate summary statistics for perfect storm signals.
        
        Args:
            signals: List of perfect storm signals
            
        Returns:
            Dict with summary statistics
        """
        try:
            if not signals:
                return {'total_signals': 0}
            
            # Signal strength statistics
            strengths = [s.signal_strength for s in signals]
            entry_scores = [s.entry_score for s in signals]
            
            # Urgency breakdown
            urgency_counts = {}
            for signal in signals:
                urgency = signal.urgency_level
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            # Sector breakdown
            sector_counts = {}
            for signal in signals:
                sector = signal.sector
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Valuation metrics
            margins_of_safety = [s.margin_of_safety for s in signals]
            vix_levels = [s.current_vix for s in signals]
            
            # Ready for trading count
            ready_count = sum(1 for s in signals if s.ready_for_options_analysis)
            
            summary = {
                'total_signals': len(signals),
                'ready_for_trading': ready_count,
                'signal_strength_avg': statistics.mean(strengths),
                'entry_score_avg': statistics.mean(entry_scores),
                'urgency_breakdown': dict(sorted(urgency_counts.items())),
                'sector_breakdown': dict(sorted(sector_counts.items(), 
                                              key=lambda x: x[1], reverse=True)),
                'avg_margin_of_safety': statistics.mean(margins_of_safety),
                'avg_vix_level': statistics.mean(vix_levels),
                'min_entry_score': min(entry_scores),
                'max_entry_score': max(entry_scores),
                'signals_above_90': sum(1 for score in entry_scores if score >= 90)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating signal summary: {e}")
            return {'total_signals': 0, 'error': str(e)}


# Convenience functions for external use
async def scan_for_opportunities(fmp_api_key: str, custom_symbols: List[str] = None) -> List[PerfectStormSignal]:
    """
    Scan for perfect storm trading opportunities.
    
    Args:
        fmp_api_key: Financial Modeling Prep API key
        custom_symbols: Optional custom symbol list
        
    Returns:
        List of perfect storm signals detected
    """
    detector = PerfectStormDetector(fmp_api_key)
    
    # Initialize watchlist
    await detector.initialize_watchlist(custom_symbols)
    
    # Scan for signals
    signals = await detector.scan_for_perfect_storms()
    
    return signals


async def quick_storm_check(symbol: str, fmp_api_key: str) -> Optional[PerfectStormSignal]:
    """
    Quick perfect storm check for individual stock.
    
    Args:
        symbol: Stock symbol to check
        fmp_api_key: FMP API key
        
    Returns:
        PerfectStormSignal if detected, None otherwise
    """
    detector = PerfectStormDetector(fmp_api_key)
    
    # Add symbol to watchlist
    await detector.initialize_watchlist([symbol])
    
    # Scan just this symbol
    signals = await detector.scan_for_perfect_storms()
    
    return signals[0] if signals else None