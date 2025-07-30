"""
Options Greeks Calculator for Holy Grail Strategy
Calculates option Greeks (Delta, Gamma, Theta, Vega) and probabilities

Used for:
- Risk assessment of options positions
- Probability calculations for strategy optimization
- Greeks-based position management
- Volatility impact analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from scipy.stats import norm

from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class OptionsGreeks:
    """Options Greeks and probability metrics"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    
    # Current pricing
    spot_price: float
    option_price: float
    implied_volatility: float
    time_to_expiry: float  # Years
    risk_free_rate: float
    
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    # Probabilities
    probability_itm: float  # In-the-money at expiration
    probability_otm: float  # Out-of-the-money at expiration
    probability_touch: float  # Probability of touching strike before expiry
    
    # Risk metrics
    intrinsic_value: float
    time_value: float
    moneyness: float  # Strike/Spot ratio


class GreeksCalculator:
    """
    Calculates option Greeks using Black-Scholes model.
    
    Provides accurate Greeks for:
    - Risk management of options positions
    - Probability analysis for strategy decisions
    - Volatility impact assessment
    - Time decay analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        """
        Initialize Greeks calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate (default 4.5%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_greeks(self, spot_price: float, strike: float, time_to_expiry: float,
                        volatility: float, option_type: str = 'put',
                        option_price: float = None) -> OptionsGreeks:
        """
        Calculate complete Greeks and probabilities for an option.
        
        Args:
            spot_price: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (decimal, e.g., 0.25 for 25%)
            option_type: 'call' or 'put'
            option_price: Current option price (calculated if None)
            
        Returns:
            OptionsGreeks object with all calculations
        """
        try:
            # Input validation
            if time_to_expiry <= 0:
                time_to_expiry = 1/365  # Minimum 1 day
            
            if volatility <= 0:
                volatility = 0.01  # Minimum 1% volatility
            
            # Calculate d1 and d2 for Black-Scholes
            d1 = self._calculate_d1(spot_price, strike, time_to_expiry, volatility, self.risk_free_rate)
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Calculate option price if not provided
            if option_price is None:
                option_price = self._black_scholes_price(
                    spot_price, strike, time_to_expiry, volatility, self.risk_free_rate, option_type
                )
            
            # Calculate Greeks
            delta = self._calculate_delta(d1, option_type)
            gamma = self._calculate_gamma(spot_price, d1, volatility, time_to_expiry)
            theta = self._calculate_theta(spot_price, strike, d1, d2, volatility, time_to_expiry, option_type)
            vega = self._calculate_vega(spot_price, d1, time_to_expiry)
            rho = self._calculate_rho(strike, d2, time_to_expiry, option_type)
            
            # Calculate probabilities
            prob_itm = self._probability_itm(d2, option_type)
            prob_otm = 1 - prob_itm
            prob_touch = self._probability_touch(spot_price, strike, volatility, time_to_expiry)
            
            # Calculate additional metrics
            if option_type.lower() == 'call':
                intrinsic_value = max(0, spot_price - strike)
            else:  # put
                intrinsic_value = max(0, strike - spot_price)
            
            time_value = option_price - intrinsic_value
            moneyness = strike / spot_price
            
            return OptionsGreeks(
                symbol="",  # To be filled by caller
                strike=strike,
                expiry="",  # To be filled by caller
                option_type=option_type,
                spot_price=spot_price,
                option_price=option_price,
                implied_volatility=volatility,
                time_to_expiry=time_to_expiry,
                risk_free_rate=self.risk_free_rate,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                probability_itm=prob_itm,
                probability_otm=prob_otm,
                probability_touch=prob_touch,
                intrinsic_value=intrinsic_value,
                time_value=time_value,
                moneyness=moneyness
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            # Return zero Greeks on error
            return self._create_zero_greeks(spot_price, strike, time_to_expiry, volatility, option_type)
    
    def _calculate_d1(self, S: float, K: float, T: float, sigma: float, r: float) -> float:
        """Calculate d1 parameter for Black-Scholes"""
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    def _black_scholes_price(self, S: float, K: float, T: float, sigma: float, 
                           r: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        d1 = self._calculate_d1(S, K, T, sigma, r)
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)
    
    def _calculate_delta(self, d1: float, option_type: str) -> float:
        """Calculate Delta (price sensitivity to underlying price change)"""
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def _calculate_gamma(self, S: float, d1: float, sigma: float, T: float) -> float:
        """Calculate Gamma (rate of change of Delta)"""
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    def _calculate_theta(self, S: float, K: float, d1: float, d2: float, 
                        sigma: float, T: float, option_type: str) -> float:
        """Calculate Theta (time decay)"""
        theta_common = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        
        if option_type.lower() == 'call':
            theta = theta_common - self.risk_free_rate * K * math.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # put
            theta = theta_common + self.risk_free_rate * K * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
        
        return theta / 365  # Convert to daily theta
    
    def _calculate_vega(self, S: float, d1: float, T: float) -> float:
        """Calculate Vega (sensitivity to volatility)"""
        return S * norm.pdf(d1) * math.sqrt(T) / 100  # Divide by 100 for 1% vol change
    
    def _calculate_rho(self, K: float, d2: float, T: float, option_type: str) -> float:
        """Calculate Rho (sensitivity to interest rate)"""
        if option_type.lower() == 'call':
            return K * T * math.exp(-self.risk_free_rate * T) * norm.cdf(d2) / 100
        else:  # put
            return -K * T * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2) / 100
    
    def _probability_itm(self, d2: float, option_type: str) -> float:
        """Calculate probability of finishing in-the-money"""
        if option_type.lower() == 'call':
            return norm.cdf(d2)
        else:  # put
            return norm.cdf(-d2)
    
    def _probability_touch(self, S: float, K: float, sigma: float, T: float) -> float:
        """Calculate probability of touching strike price before expiration"""
        if S == K:
            return 1.0
        
        # Probability of touching barrier (approximate)
        mu = self.risk_free_rate - 0.5 * sigma ** 2
        barrier_ratio = math.log(K / S)
        
        prob_touch = 2 * norm.cdf(-abs(barrier_ratio) / (sigma * math.sqrt(T)))
        
        return min(max(prob_touch, 0.0), 1.0)
    
    def _create_zero_greeks(self, S: float, K: float, T: float, sigma: float, option_type: str) -> OptionsGreeks:
        """Create OptionsGreeks object with zero values for error cases"""
        return OptionsGreeks(
            symbol="",
            strike=K,
            expiry="",
            option_type=option_type,
            spot_price=S,
            option_price=0,
            implied_volatility=sigma,
            time_to_expiry=T,
            risk_free_rate=self.risk_free_rate,
            delta=0,
            gamma=0,
            theta=0,
            vega=0,
            rho=0,
            probability_itm=0,
            probability_otm=1,
            probability_touch=0,
            intrinsic_value=0,
            time_value=0,
            moneyness=K/S if S > 0 else 1
        )
    
    def calculate_implied_volatility(self, market_price: float, spot_price: float, 
                                   strike: float, time_to_expiry: float, 
                                   option_type: str = 'put') -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Current market price of option
            spot_price: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility as decimal
        """
        try:
            # Initial guess
            iv = 0.25  # 25%
            tolerance = 1e-6
            max_iterations = 100
            
            for _ in range(max_iterations):
                # Calculate theoretical price and vega
                theoretical_price = self._black_scholes_price(
                    spot_price, strike, time_to_expiry, iv, self.risk_free_rate, option_type
                )
                
                d1 = self._calculate_d1(spot_price, strike, time_to_expiry, iv, self.risk_free_rate)
                vega = spot_price * norm.pdf(d1) * math.sqrt(time_to_expiry)
                
                # Price difference
                price_diff = theoretical_price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    return iv
                
                # Newton-Raphson update
                if vega != 0:
                    iv = iv - price_diff / vega
                    iv = max(0.001, min(iv, 5.0))  # Keep IV between 0.1% and 500%
                else:
                    break
            
            return iv
            
        except Exception as e:
            logger.debug(f"Error calculating implied volatility: {e}")
            return 0.25  # Default to 25%
    
    def portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks for multiple positions.
        
        Args:
            positions: List of position dicts with Greeks data
            
        Returns:
            Dict with portfolio Greeks
        """
        try:
            portfolio_greeks = {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            
            for position in positions:
                quantity = position.get('quantity', 0)
                multiplier = position.get('multiplier', 100)  # Options: 100 shares per contract
                
                # Add weighted Greeks
                portfolio_greeks['delta'] += quantity * multiplier * position.get('delta', 0)
                portfolio_greeks['gamma'] += quantity * multiplier * position.get('gamma', 0)
                portfolio_greeks['theta'] += quantity * multiplier * position.get('theta', 0)
                portfolio_greeks['vega'] += quantity * multiplier * position.get('vega', 0)
                portfolio_greeks['rho'] += quantity * multiplier * position.get('rho', 0)
            
            return portfolio_greeks
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def days_to_expiry(self, expiry_date: str) -> float:
        """Convert expiry date string to days (as fraction of year)"""
        try:
            if len(expiry_date) == 8:  # YYYYMMDD format
                exp_date = datetime.strptime(expiry_date, '%Y%m%d')
            else:
                exp_date = datetime.strptime(expiry_date, '%Y-%m-%d')
            
            days = (exp_date - datetime.now()).days
            return max(days / 365.0, 1/365)  # Minimum 1 day
            
        except Exception as e:
            logger.debug(f"Error parsing expiry date {expiry_date}: {e}")
            return 30/365  # Default to 30 days


# Convenience functions
def calculate_put_greeks(spot_price: float, strike: float, days_to_expiry: int,
                        implied_vol: float, option_price: float = None) -> OptionsGreeks:
    """
    Quick Greeks calculation for put options.
    
    Args:
        spot_price: Current stock price
        strike: Put strike price
        days_to_expiry: Days until expiration
        implied_vol: Implied volatility (decimal)
        option_price: Current option price
        
    Returns:
        OptionsGreeks object
    """
    calculator = GreeksCalculator()
    time_to_expiry = days_to_expiry / 365.0
    
    return calculator.calculate_greeks(
        spot_price=spot_price,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=implied_vol,
        option_type='put',
        option_price=option_price
    )


def quick_probability_otm(spot_price: float, strike: float, days_to_expiry: int,
                         implied_vol: float) -> float:
    """
    Quick calculation of probability of put expiring out-of-the-money.
    
    Args:
        spot_price: Current stock price
        strike: Put strike price
        days_to_expiry: Days until expiration
        implied_vol: Implied volatility
        
    Returns:
        Probability of expiring OTM (0-1)
    """
    greeks = calculate_put_greeks(spot_price, strike, days_to_expiry, implied_vol)
    return greeks.probability_otm