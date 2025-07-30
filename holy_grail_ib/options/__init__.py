# Create these __init__.py files in their respective directories:

# holy_grail_ib/__init__.py
"""
Holy Grail Options Strategy Package
Complete implementation of Adam Khoo's cash-secured put selling system
"""

__version__ = "1.0.0"
__author__ = "Holy Grail Strategy Team"

# config/__init__.py
"""Configuration module for Holy Grail strategy"""

from .criteria import get_criteria, CRITERIA

__all__ = ['get_criteria', 'CRITERIA']

# data/__init__.py
"""Data providers and clients"""

from .ib_client import get_ib_client, IBClient
from .fmp_client import get_fmp_client, FMPClient

__all__ = ['get_ib_client', 'IBClient', 'get_fmp_client', 'FMPClient']

# screening/__init__.py
"""Stock screening and valuation modules"""

from .business_quality import BusinessQualityScreener, screen_sp500_for_quality
from .valuation import DCFValuationEngine, calculate_fair_value

__all__ = ['BusinessQualityScreener', 'screen_sp500_for_quality', 'DCFValuationEngine', 'calculate_fair_value']

# signals/__init__.py
"""Signal detection modules"""

from .perfect_storm import PerfectStormDetector, scan_for_opportunities

__all__ = ['PerfectStormDetector', 'scan_for_opportunities']

# options/__init__.py
"""Options analysis modules"""

from .chain_analyzer import OptionsChainAnalyzer, analyze_signal_options

__all__ = ['OptionsChainAnalyzer', 'analyze_signal_options']

# trading/__init__.py
"""Trading execution and risk management"""

from .order_manager import OrderManager, quick_put_sale
from .risk_manager import RiskManager, check_position_risk

__all__ = ['OrderManager', 'quick_put_sale', 'RiskManager', 'check_position_risk']

# strategy/__init__.py
"""High-level strategy orchestration"""

from .entry_engine import HolyGrailEntryEngine, run_holy_grail_scan

__all__ = ['HolyGrailEntryEngine', 'run_holy_grail_scan']

# alerts/__init__.py
"""Notification and alerting system"""

from .notification import NotificationManager, send_opportunity_alert

__all__ = ['NotificationManager', 'send_opportunity_alert']

# utils/__init__.py
"""Utility functions and helpers"""

from .logging_setup import setup_logging, get_logger

__all__ = ['setup_logging', 'get_logger']