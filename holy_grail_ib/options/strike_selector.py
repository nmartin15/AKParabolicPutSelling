"""
Strike Selection Engine for Holy Grail Strategy
Intelligently selects optimal put strikes based on Adam's criteria

Factors considered:
- Distance from current price (8-15% OTM)
- Relationship to intrinsic value (≤105% of DCF value)
- Premium attractiveness (≥2% of strike)
- Probability of profit (≥70% OTM)
- Risk/reward optimization
- Support level analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statistics

from config.criteria import get_criteria
from options.greeks_calculator import GreeksCalculator, calculate_put_greeks
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class StrikeCandidate:
    """Potential strike price candidate"""
    strike: float
    distance_otm_pct: float
    premium_estimate: float
    premium_pct: float
    delta: float
    probability_otm: float
    
    # Scoring components
    distance_score: float
    premium_score: float
    probability_score: float
    support_score: float
    total_score: float
    
    # Risk metrics
    max_loss: float
    breakeven_price: float
    risk_reward_ratio: float
    
    # Qualitative assessment
    meets_criteria: bool
    recommendation: str  # "excellent", "good", "acceptable", "avoid"
    notes: List[str]


@dataclass
class StrikeSelection:
    """Complete strike selection analysis"""
    symbol: str
    current_price: float
    intrinsic_value: float
    support_levels: List[float]
    
    # Strike candidates
    all_candidates: List[StrikeCandidate]
    qualifying_candidates: List[StrikeCandidate]
    
    # Recommendations
    optimal_strike: Optional[StrikeCandidate]
    conservative_strike: Optional[StrikeCandidate]
    aggressive_strike: Optional[StrikeCandidate]
    
    # Analysis metadata
    selection_confidence: str  # "high", "medium", "low"
    market_conditions: str
    selection_notes: List[str]


class StrikeSelector:
    """
    Intelligent strike selection engine for cash-secured put selling.
    
    Implements Adam's methodology for finding optimal strikes that balance:
    - High probability of profit
    - Attractive premium collection
    - Reasonable assignment risk
    - Support level considerations
    """
    
    def __init__(self):
        """Initialize strike selector with criteria"""
        self.options_criteria = get_criteria('options_selection')
        self.valuation_criteria = get_criteria('valuation')
        self.greeks_calculator = GreeksCalculator()
        
    def select_optimal_strikes(self, symbol: str, current_price: float, 
                             intrinsic_value: float, support_levels: List[float],
                             days_to_expiry: int = 35, implied_vol: float = 0.30) -> StrikeSelection:
        """
        Select optimal put strikes for cash-secured put selling.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            intrinsic_value: DCF intrinsic value
            support_levels: List of technical support levels
            days_to_expiry: Days to option expiration
            implied_vol: Estimated implied volatility
            
        Returns:
            StrikeSelection with analysis and recommendations
        """
        try:
            logger.debug(f"Selecting optimal strikes for {symbol}: ${current_price:.2f} current, ${intrinsic_value:.2f} intrinsic")
            
            # Generate strike candidates
            strike_candidates = self._generate_strike_candidates(
                current_price, intrinsic_value, days_to_expiry, implied_vol
            )
            
            # Score each candidate
            scored_candidates = []
            for candidate in strike_candidates:
                scored_candidate = self._score_strike_candidate(
                    candidate, current_price, intrinsic_value, support_levels
                )
                scored_candidates.append(scored_candidate)
            
            # Filter qualifying candidates
            qualifying_candidates = [
                candidate for candidate in scored_candidates
                if candidate.meets_criteria
            ]
            
            # Sort by total score
            qualifying_candidates.sort(key=lambda x: x.total_score, reverse=True)
            scored_candidates.sort(key=lambda x: x.total_score, reverse=True)
            
            # Select recommendations
            optimal_strike = qualifying_candidates[0] if qualifying_candidates else None
            conservative_strike = self._select_conservative_strike(qualifying_candidates)
            aggressive_strike = self._select_aggressive_strike(qualifying_candidates)
            
            # Assess selection confidence
            confidence = self._assess_selection_confidence(qualifying_candidates, support_levels)
            market_conditions = self._assess_market_conditions(implied_vol, current_price, intrinsic_value)
            
            # Generate selection notes
            selection_notes = self._generate_selection_notes(
                qualifying_candidates, current_price, intrinsic_value, support_levels
            )
            
            selection = StrikeSelection(
                symbol=symbol,
                current_price=current_price,
                intrinsic_value=intrinsic_value,
                support_levels=support_levels,
                all_candidates=scored_candidates,
                qualifying_candidates=qualifying_candidates,
                optimal_strike=optimal_strike,
                conservative_strike=conservative_strike,
                aggressive_strike=aggressive_strike,
                selection_confidence=confidence,
                market_conditions=market_conditions,
                selection_notes=selection_notes
            )
            
            logger.debug(f"Strike selection complete: {len(qualifying_candidates)} qualifying strikes found")
            
            return selection
            
        except Exception as e:
            logger.error(f"Error selecting optimal strikes for {symbol}: {e}")
            return self._create_empty_selection(symbol, current_price, intrinsic_value)
    
    def _generate_strike_candidates(self, current_price: float, intrinsic_value: float,
                                  days_to_expiry: int, implied_vol: float) -> List[StrikeCandidate]:
        """Generate potential strike candidates within criteria range"""
        candidates = []
        
        try:
            # Define strike range based on criteria (8-15% OTM)
            min_distance = self.options_criteria.min_otm_distance
            max_distance = self.options_criteria.max_otm_distance
            
            min_strike = current_price * (1 - max_distance)
            max_strike = current_price * (1 - min_distance)
            
            # Also ensure strikes don't exceed intrinsic value limit
            max_strike = min(max_strike, intrinsic_value * self.options_criteria.max_strike_to_fair_value)
            
            # Generate strikes in $2.50 increments (typical options spacing)
            strike_increment = 2.5
            current_strike = math.floor(min_strike / strike_increment) * strike_increment
            
            while current_strike <= max_strike:
                distance_otm = (current_price - current_strike) / current_price
                
                # Estimate premium using simplified Black-Scholes
                greeks = calculate_put_greeks(
                    spot_price=current_price,
                    strike=current_strike,
                    days_to_expiry=days_to_expiry,
                    implied_vol=implied_vol
                )
                
                premium_estimate = max(greeks.intrinsic_value, 
                                     self._estimate_option_premium(current_price, current_strike, 
                                                                  days_to_expiry, implied_vol))
                
                premium_pct = premium_estimate / current_strike
                
                candidate = StrikeCandidate(
                    strike=current_strike,
                    distance_otm_pct=distance_otm,
                    premium_estimate=premium_estimate,
                    premium_pct=premium_pct,
                    delta=greeks.delta,
                    probability_otm=greeks.probability_otm,
                    distance_score=0,  # Will be calculated in scoring
                    premium_score=0,
                    probability_score=0,
                    support_score=0,
                    total_score=0,
                    max_loss=current_strike - premium_estimate,
                    breakeven_price=current_strike - premium_estimate,
                    risk_reward_ratio=premium_estimate / (current_strike - premium_estimate) if current_strike > premium_estimate else 0,
                    meets_criteria=False,  # Will be determined in scoring
                    recommendation="pending",
                    notes=[]
                )
                
                candidates.append(candidate)
                current_strike += strike_increment
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error generating strike candidates: {e}")
            return []
    
    def _estimate_option_premium(self, spot: float, strike: float, days: int, vol: float) -> float:
        """Estimate option premium using simplified model"""
        try:
            time_to_expiry = days / 365.0
            
            # Simplified premium estimation
            intrinsic = max(0, strike - spot)
            time_value = vol * spot * math.sqrt(time_to_expiry) * 0.4  # Approximation
            
            return intrinsic + time_value
            
        except Exception as e:
            logger.debug(f"Error estimating option premium: {e}")
            return 0.0
    
    def _score_strike_candidate(self, candidate: StrikeCandidate, current_price: float,
                               intrinsic_value: float, support_levels: List[float]) -> StrikeCandidate:
        """Score strike candidate based on multiple factors"""
        try:
            # Distance Score (25 points) - prefer 10-12% OTM
            optimal_distance = 0.11  # 11% OTM is sweet spot
            distance_deviation = abs(candidate.distance_otm_pct - optimal_distance)
            candidate.distance_score = max(0, 25 - (distance_deviation * 500))  # Penalty for deviation
            
            # Premium Score (30 points) - prefer higher premiums
            if candidate.premium_pct >= 0.030:  # ≥3%
                candidate.premium_score = 30
            elif candidate.premium_pct >= 0.025:  # ≥2.5%
                candidate.premium_score = 25
            elif candidate.premium_pct >= 0.020:  # ≥2%
                candidate.premium_score = 20
            elif candidate.premium_pct >= 0.015:  # ≥1.5%
                candidate.premium_score = 10
            else:
                candidate.premium_score = 0
            
            # Probability Score (25 points) - prefer high probability OTM
            if candidate.probability_otm >= 0.85:  # ≥85%
                candidate.probability_score = 25
            elif candidate.probability_otm >= 0.75:  # ≥75%
                candidate.probability_score = 20
            elif candidate.probability_otm >= 0.70:  # ≥70%
                candidate.probability_score = 15
            elif candidate.probability_otm >= 0.60:  # ≥60%
                candidate.probability_score = 10
            else:
                candidate.probability_score = 0
            
            # Support Score (20 points) - proximity to support levels
            candidate.support_score = self._calculate_support_score(candidate.strike, support_levels)
            
            # Total Score
            candidate.total_score = (candidate.distance_score + candidate.premium_score + 
                                   candidate.probability_score + candidate.support_score)
            
            # Check if meets basic criteria
            candidate.meets_criteria = self._check_basic_criteria(candidate, intrinsic_value)
            
            # Generate recommendation
            candidate.recommendation = self._generate_recommendation(candidate)
            
            # Add notes
            candidate.notes = self._generate_candidate_notes(candidate, current_price, intrinsic_value)
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error scoring strike candidate: {e}")
            return candidate
    
    def _calculate_support_score(self, strike: float, support_levels: List[float]) -> float:
        """Calculate score based on proximity to support levels"""
        if not support_levels:
            return 10  # Neutral score if no support data
        
        try:
            # Find closest support level
            distances = [abs(strike - support) for support in support_levels]
            min_distance = min(distances)
            
            # Score based on proximity (closer = higher score)
            if min_distance <= strike * 0.02:  # Within 2%
                return 20
            elif min_distance <= strike * 0.05:  # Within 5%
                return 15
            elif min_distance <= strike * 0.10:  # Within 10%
                return 10
            else:
                return 5
                
        except Exception as e:
            logger.debug(f"Error calculating support score: {e}")
            return 10
    
    def _check_basic_criteria(self, candidate: StrikeCandidate, intrinsic_value: float) -> bool:
        """Check if candidate meets Adam's basic criteria"""
        try:
            # Premium percentage check
            if candidate.premium_pct < self.options_criteria.min_premium_pct:
                return False
            
            # Probability OTM check
            if candidate.probability_otm < self.options_criteria.min_prob_otm:
                return False
            
            # Delta range check
            if not (self.options_criteria.min_delta <= candidate.delta <= self.options_criteria.max_delta):
                return False
            
            # Strike vs intrinsic value check
            if candidate.strike > intrinsic_value * self.options_criteria.max_strike_to_fair_value:
                return False
            
            # Distance OTM check
            if not (self.options_criteria.min_otm_distance <= candidate.distance_otm_pct <= self.options_criteria.max_otm_distance):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking basic criteria: {e}")
            return False
    
    def _generate_recommendation(self, candidate: StrikeCandidate) -> str:
        """Generate qualitative recommendation for candidate"""
        if not candidate.meets_criteria:
            return "avoid"
        
        if candidate.total_score >= 80:
            return "excellent"
        elif candidate.total_score >= 65:
            return "good"
        elif candidate.total_score >= 50:
            return "acceptable"
        else:
            return "marginal"
    
    def _generate_candidate_notes(self, candidate: StrikeCandidate, 
                                 current_price: float, intrinsic_value: float) -> List[str]:
        """Generate explanatory notes for candidate"""
        notes = []
        
        try:
            # Premium notes
            if candidate.premium_pct >= 0.025:
                notes.append(f"Attractive premium: {candidate.premium_pct:.1%}")
            elif candidate.premium_pct < 0.015:
                notes.append(f"Low premium: {candidate.premium_pct:.1%}")
            
            # Probability notes
            if candidate.probability_otm >= 0.80:
                notes.append(f"High probability of profit: {candidate.probability_otm:.0%}")
            elif candidate.probability_otm < 0.70:
                notes.append(f"Lower probability of profit: {candidate.probability_otm:.0%}")
            
            # Distance notes
            if candidate.distance_otm_pct > 0.13:
                notes.append("Far out-of-the-money - lower assignment risk")
            elif candidate.distance_otm_pct < 0.09:
                notes.append("Close to current price - higher assignment risk")
            
            # Value notes
            if candidate.strike > intrinsic_value:
                notes.append(f"Strike above intrinsic value (${intrinsic_value:.2f})")
            else:
                value_discount = (intrinsic_value - candidate.strike) / intrinsic_value
                notes.append(f"Good value: {value_discount:.0%} below intrinsic value")
            
            # Risk/reward notes
            if candidate.risk_reward_ratio > 0.15:
                notes.append("Excellent risk/reward ratio")
            elif candidate.risk_reward_ratio < 0.05:
                notes.append("Poor risk/reward ratio")
            
            return notes
            
        except Exception as e:
            logger.debug(f"Error generating candidate notes: {e}")
            return ["Analysis error"]
    
    def _select_conservative_strike(self, candidates: List[StrikeCandidate]) -> Optional[StrikeCandidate]:
        """Select most conservative strike (highest probability OTM)"""
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x.probability_otm)
    
    def _select_aggressive_strike(self, candidates: List[StrikeCandidate]) -> Optional[StrikeCandidate]:
        """Select most aggressive strike (highest premium)"""
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x.premium_pct)
    
    def _assess_selection_confidence(self, candidates: List[StrikeCandidate], 
                                   support_levels: List[float]) -> str:
        """Assess confidence in strike selection"""
        try:
            if not candidates:
                return "low"
            
            top_candidate = candidates[0]
            
            # High confidence criteria
            if (top_candidate.total_score >= 75 and
                top_candidate.probability_otm >= 0.75 and
                top_candidate.premium_pct >= 0.020 and
                len(support_levels) >= 2):
                return "high"
            
            # Medium confidence criteria
            if (top_candidate.total_score >= 60 and
                top_candidate.probability_otm >= 0.70 and
                top_candidate.premium_pct >= 0.015):
                return "medium"
            
            return "low"
            
        except Exception as e:
            logger.debug(f"Error assessing selection confidence: {e}")
            return "low"
    
    def _assess_market_conditions(self, implied_vol: float, current_price: float, 
                                intrinsic_value: float) -> str:
        """Assess current market conditions for options selling"""
        try:
            # Volatility assessment
            if implied_vol >= 0.40:
                vol_condition = "high_vol"
            elif implied_vol >= 0.25:
                vol_condition = "normal_vol"
            else:
                vol_condition = "low_vol"
            
            # Valuation assessment
            price_to_value = current_price / intrinsic_value
            if price_to_value <= 0.90:
                value_condition = "undervalued"
            elif price_to_value <= 1.05:
                value_condition = "fair_value"
            else:
                value_condition = "overvalued"
            
            # Combined assessment
            if vol_condition == "high_vol" and value_condition in ["undervalued", "fair_value"]:
                return "excellent"
            elif vol_condition == "normal_vol" and value_condition == "undervalued":
                return "good"
            elif vol_condition == "low_vol" or value_condition == "overvalued":
                return "poor"
            else:
                return "fair"
                
        except Exception as e:
            logger.debug(f"Error assessing market conditions: {e}")
            return "unknown"
    
    def _generate_selection_notes(self, candidates: List[StrikeCandidate],
                                 current_price: float, intrinsic_value: float,
                                 support_levels: List[float]) -> List[str]:
        """Generate overall selection notes"""
        notes = []
        
        try:
            if not candidates:
                notes.append("No strikes meet minimum criteria")
                return notes
            
            notes.append(f"Analyzed {len(candidates)} qualifying strikes")
            
            # Best candidate notes
            best = candidates[0]
            notes.append(f"Top strike: ${best.strike:.0f} with {best.total_score:.0f}/100 score")
            notes.append(f"Premium: {best.premium_pct:.1%}, Probability: {best.probability_otm:.0%}")
            
            # Support level notes
            if support_levels:
                closest_support = min(support_levels, key=lambda x: abs(x - best.strike))
                support_distance = abs(best.strike - closest_support) / best.strike
                notes.append(f"Closest support: ${closest_support:.2f} ({support_distance:.1%} away)")
            
            # Value notes
            value_relationship = best.strike / intrinsic_value
            notes.append(f"Strike is {value_relationship:.0%} of intrinsic value")
            
            return notes
            
        except Exception as e:
            logger.debug(f"Error generating selection notes: {e}")
            return ["Analysis completed with errors"]
    
    def _create_empty_selection(self, symbol: str, current_price: float, 
                               intrinsic_value: float) -> StrikeSelection:
        """Create empty selection for error cases"""
        return StrikeSelection(
            symbol=symbol,
            current_price=current_price,
            intrinsic_value=intrinsic_value,
            support_levels=[],
            all_candidates=[],
            qualifying_candidates=[],
            optimal_strike=None,
            conservative_strike=None,
            aggressive_strike=None,
            selection_confidence="low",
            market_conditions="unknown",
            selection_notes=["Error in strike selection analysis"]
        )


# Convenience functions
def select_best_put_strike(symbol: str, current_price: float, intrinsic_value: float,
                          support_levels: List[float] = None, days_to_expiry: int = 35,
                          implied_vol: float = 0.30) -> Optional[StrikeCandidate]:
    """
    Quick selection of best put strike for a stock.
    
    Args:
        symbol: Stock symbol
        current_price: Current stock price
        intrinsic_value: DCF intrinsic value
        support_levels: Technical support levels
        days_to_expiry: Days to option expiration
        implied_vol: Implied volatility estimate
        
    Returns:
        Best StrikeCandidate or None
    """
    selector = StrikeSelector()
    selection = selector.select_optimal_strikes(
        symbol=symbol,
        current_price=current_price,
        intrinsic_value=intrinsic_value,
        support_levels=support_levels or [],
        days_to_expiry=days_to_expiry,
        implied_vol=implied_vol
    )
    
    return selection.optimal_strike


def analyze_strike_range(current_price: float, intrinsic_value: float,
                        min_premium_pct: float = 0.02) -> Tuple[float, float]:
    """
    Calculate optimal strike range for put selling.
    
    Args:
        current_price: Current stock price
        intrinsic_value: DCF intrinsic value
        min_premium_pct: Minimum premium percentage
        
    Returns:
        Tuple of (min_strike, max_strike)
    """
    criteria = get_criteria('options_selection')
    
    # Range based on distance OTM
    min_strike_distance = current_price * (1 - criteria.max_otm_distance)
    max_strike_distance = current_price * (1 - criteria.min_otm_distance)
    
    # Range based on intrinsic value
    max_strike_value = intrinsic_value * criteria.max_strike_to_fair_value
    
    # Final range
    min_strike = min_strike_distance
    max_strike = min(max_strike_distance, max_strike_value)
    
    return min_strike, max_strike