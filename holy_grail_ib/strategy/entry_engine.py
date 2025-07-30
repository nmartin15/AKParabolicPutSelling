"""
Entry Engine for Holy Grail Options Strategy
Orchestrates the complete workflow from screening to trade execution

This is the main engine that coordinates all components:
1. Business quality screening
2. DCF valuation 
3. Perfect storm detection
4. Options chain analysis
5. Risk management
6. Trade execution
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from config.criteria import get_criteria
from screening.business_quality import BusinessQualityScreener, QualifiedCompany
from screening.valuation import DCFValuationEngine, DCFOutputs
from signals.perfect_storm import PerfectStormDetector, PerfectStormSignal
from options.chain_analyzer import OptionsChainAnalyzer, OptionsAnalysis
from trading.risk_manager import RiskManager
from trading.order_manager import OrderManager
from alerts.notification import NotificationManager
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TradeOpportunity:
    """Complete trade opportunity ready for execution"""
    symbol: str
    company_name: str
    
    # Signal components
    perfect_storm_signal: PerfectStormSignal
    options_analysis: OptionsAnalysis
    
    # Trade details
    recommended_action: str  # "execute", "monitor", "skip"
    put_strike: float
    put_expiry: str
    premium_target: float
    contracts_to_sell: int
    capital_required: float
    
    # Risk metrics
    max_loss: float
    probability_profit: float
    annualized_return: float
    portfolio_impact: float
    
    # Execution readiness
    execution_priority: int  # 1-5 (5 = highest)
    time_sensitivity: str   # "immediate", "today", "this_week"
    market_conditions: str
    
    # Warnings and notes
    trade_warnings: List[str]
    execution_notes: List[str]


@dataclass
class StrategyPerformance:
    """Strategy performance tracking"""
    total_opportunities_found: int
    trades_executed: int
    trades_profitable: int
    total_premium_collected: float
    total_assignments: int
    current_open_positions: int
    portfolio_allocation: float
    
    # Performance metrics
    win_rate: float
    average_return_per_trade: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float


class HolyGrailEntryEngine:
    """
    Main orchestrating engine for Adam's Holy Grail strategy.
    
    Coordinates all components to identify and execute optimal 
    cash-secured put selling opportunities.
    """
    
    def __init__(self, fmp_api_key: str, portfolio_value: float = 1000000,
                 ib_host: str = '127.0.0.1', ib_port: int = 7497):
        """
        Initialize the Holy Grail entry engine.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key
            portfolio_value: Total portfolio value
            ib_host: Interactive Brokers host
            ib_port: Interactive Brokers port
        """
        self.fmp_api_key = fmp_api_key
        self.portfolio_value = portfolio_value
        self.ib_host = ib_host
        self.ib_port = ib_port
        
        # Initialize component engines
        self.quality_screener = BusinessQualityScreener(fmp_api_key)
        self.valuation_engine = DCFValuationEngine(fmp_api_key)
        self.storm_detector = PerfectStormDetector(fmp_api_key, ib_host, ib_port)
        self.options_analyzer = OptionsChainAnalyzer(ib_host, ib_port)
        self.risk_manager = RiskManager(portfolio_value)
        self.order_manager = OrderManager(ib_host, ib_port)
        self.notification_manager = NotificationManager()
        
        # Strategy state
        self.watchlist_initialized = False
        self.last_full_scan: Optional[datetime] = None
        self.active_opportunities: List[TradeOpportunity] = []
        self.executed_trades: List[TradeOpportunity] = []
        self.performance: StrategyPerformance = StrategyPerformance(
            total_opportunities_found=0,
            trades_executed=0,
            trades_profitable=0,
            total_premium_collected=0.0,
            total_assignments=0,
            current_open_positions=0,
            portfolio_allocation=0.0,
            win_rate=0.0,
            average_return_per_trade=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0
        )
        
        # Load criteria
        self.risk_criteria = get_criteria('risk_management')
        self.scoring_weights = get_criteria('scoring_weights')
    
    async def initialize_strategy(self, custom_symbols: List[str] = None) -> bool:
        """
        Initialize the strategy with quality screening and valuations.
        
        Args:
            custom_symbols: Optional custom universe, otherwise S&P 500
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Holy Grail strategy...")
            
            # Initialize watchlist with quality screening + valuations
            watchlist_size = await self.storm_detector.initialize_watchlist(custom_symbols)
            
            if watchlist_size == 0:
                logger.error("No qualified companies found in universe")
                return False
            
            # Initialize risk manager with current portfolio
            await self.risk_manager.update_portfolio_state()
            
            self.watchlist_initialized = True
            logger.info(f"Strategy initialized with {watchlist_size} qualified companies")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            return False
    
    async def run_full_scan(self, execute_trades: bool = False) -> List[TradeOpportunity]:
        """
        Run complete scan for trading opportunities.
        
        Args:
            execute_trades: Whether to automatically execute qualifying trades
            
        Returns:
            List of trade opportunities found
        """
        try:
            if not self.watchlist_initialized:
                logger.warning("Strategy not initialized. Call initialize_strategy() first.")
                return []
            
            logger.info("Running full Holy Grail opportunity scan...")
            
            # Step 1: Scan for perfect storm signals
            storm_signals = await self.storm_detector.scan_for_perfect_storms()
            logger.info(f"Perfect storm scan: {len(storm_signals)} signals detected")
            
            if not storm_signals:
                logger.info("No perfect storm signals found")
                return []
            
            # Step 2: Analyze options for each signal
            opportunities = []
            for signal in storm_signals:
                try:
                    # Get options analysis
                    options_analysis = await self.options_analyzer.analyze_perfect_storm_options(
                        signal, self.portfolio_value
                    )
                    
                    # Create trade opportunity
                    opportunity = await self._create_trade_opportunity(signal, options_analysis)
                    
                    if opportunity and opportunity.recommended_action in ["execute", "monitor"]:
                        opportunities.append(opportunity)
                        logger.info(f"Trade opportunity: {opportunity.symbol} - {opportunity.recommended_action}")
                    
                except Exception as e:
                    logger.debug(f"Error analyzing options for {signal.symbol}: {e}")
                    continue
            
            # Step 3: Risk management and prioritization
            validated_opportunities = await self._validate_and_prioritize_opportunities(opportunities)
            
            # Step 4: Execute trades if requested
            if execute_trades and validated_opportunities:
                executed_count = await self._execute_qualified_trades(validated_opportunities)
                logger.info(f"Executed {executed_count} trades automatically")
            
            # Update performance tracking
            self.performance.total_opportunities_found += len(validated_opportunities)
            self.last_full_scan = datetime.now()
            self.active_opportunities = validated_opportunities
            
            logger.info(f"Full scan complete: {len(validated_opportunities)} opportunities found")
            
            # Send notifications for high-priority opportunities
            await self._send_opportunity_notifications(validated_opportunities)
            
            return validated_opportunities
            
        except Exception as e:
            logger.error(f"Error in full scan: {e}")
            return []
    
    async def _create_trade_opportunity(self, signal: PerfectStormSignal, 
                                       options_analysis: OptionsAnalysis) -> Optional[TradeOpportunity]:
        """
        Create trade opportunity from signal and options analysis.
        
        Args:
            signal: Perfect storm signal
            options_analysis: Options chain analysis
            
        Returns:
            TradeOpportunity if viable, None otherwise
        """
        try:
            # Check if we have a viable option to trade
            best_put = options_analysis.best_put
            if not best_put or not best_put.meets_all_criteria:
                return None
            
            # Determine recommended action
            if (signal.entry_score >= 95 and
                best_put.trade_recommendation == "strong_buy" and
                options_analysis.trade_confidence == "high"):
                recommended_action = "execute"
                execution_priority = 5
                time_sensitivity = "immediate"
            elif (signal.entry_score >= 90 and
                  best_put.trade_recommendation in ["strong_buy", "buy"] and
                  options_analysis.trade_confidence in ["high", "medium"]):
                recommended_action = "execute"
                execution_priority = 4
                time_sensitivity = "today"
            elif signal.entry_score >= 85:
                recommended_action = "monitor"
                execution_priority = 3
                time_sensitivity = "this_week"
            else:
                recommended_action = "skip"
                execution_priority = 1
                time_sensitivity = "low"
            
            # Calculate trade metrics
            contracts = options_analysis.recommended_contracts
            capital_required = options_analysis.capital_required
            max_loss = best_put.max_loss * contracts * 100  # Per contract basis
            probability_profit = best_put.probability_otm
            annualized_return = best_put.annualized_return
            portfolio_impact = options_analysis.portfolio_allocation_pct
            
            # Compile warnings
            trade_warnings = []
            trade_warnings.extend(signal.signal_warnings)
            trade_warnings.extend(options_analysis.analysis_warnings)
            trade_warnings.extend(best_put.criteria_violations)
            
            # Execution notes
            execution_notes = [
                f"Entry score: {signal.entry_score}/100",
                f"Signal strength: {signal.signal_strength}/100",
                f"Trade confidence: {options_analysis.trade_confidence}",
                f"Market conditions: {options_analysis.market_conditions}",
                f"Urgency: {signal.urgency_level}"
            ]
            
            opportunity = TradeOpportunity(
                symbol=signal.symbol,
                company_name=signal.company_name,
                perfect_storm_signal=signal,
                options_analysis=options_analysis,
                recommended_action=recommended_action,
                put_strike=best_put.strike,
                put_expiry=best_put.expiry,
                premium_target=best_put.mid_price,
                contracts_to_sell=contracts,
                capital_required=capital_required,
                max_loss=max_loss,
                probability_profit=probability_profit,
                annualized_return=annualized_return,
                portfolio_impact=portfolio_impact,
                execution_priority=execution_priority,
                time_sensitivity=time_sensitivity,
                market_conditions=options_analysis.market_conditions,
                trade_warnings=trade_warnings,
                execution_notes=execution_notes
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error creating trade opportunity for {signal.symbol}: {e}")
            return None
    
    async def _validate_and_prioritize_opportunities(self, opportunities: List[TradeOpportunity]) -> List[TradeOpportunity]:
        """
        Validate opportunities against risk management and prioritize.
        
        Args:
            opportunities: List of trade opportunities
            
        Returns:
            Validated and prioritized opportunities
        """
        try:
            validated_opportunities = []
            
            for opportunity in opportunities:
                # Risk management validation
                risk_check = await self.risk_manager.validate_new_position(
                    opportunity.symbol,
                    opportunity.capital_required,
                    opportunity.portfolio_impact
                )
                
                if not risk_check['approved']:
                    opportunity.recommended_action = "skip"
                    opportunity.trade_warnings.extend(risk_check['reasons'])
                    logger.info(f"{opportunity.symbol}: Risk check failed - {risk_check['reasons']}")
                    continue
                
                # Portfolio concentration check
                if await self._check_portfolio_concentration(opportunity):
                    validated_opportunities.append(opportunity)
                else:
                    opportunity.recommended_action = "skip"
                    opportunity.trade_warnings.append("Portfolio concentration limits exceeded")
            
            # Sort by execution priority
            validated_opportunities.sort(key=lambda x: x.execution_priority, reverse=True)
            
            return validated_opportunities
            
        except Exception as e:
            logger.error(f"Error validating opportunities: {e}")
            return opportunities
    
    async def _check_portfolio_concentration(self, opportunity: TradeOpportunity) -> bool:
        """Check if opportunity violates portfolio concentration limits"""
        try:
            # Check sector concentration
            sector = opportunity.perfect_storm_signal.quality_data.sector
            current_sector_allocation = await self.risk_manager.get_sector_allocation(sector)
            
            if current_sector_allocation + opportunity.portfolio_impact > self.risk_criteria.max_sector_concentration_pct:
                return False
            
            # Check total options exposure
            current_options_allocation = await self.risk_manager.get_options_allocation()
            
            if current_options_allocation + opportunity.portfolio_impact > self.risk_criteria.max_total_options_pct:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking portfolio concentration: {e}")
            return True  # Allow trade if check fails
    
    async def _execute_qualified_trades(self, opportunities: List[TradeOpportunity]) -> int:
        """
        Execute trades for qualified opportunities.
        
        Args:
            opportunities: List of validated opportunities
            
        Returns:
            Number of trades executed
        """
        executed_count = 0
        
        try:
            # Only execute highest priority opportunities
            execution_candidates = [opp for opp in opportunities 
                                  if opp.recommended_action == "execute" and 
                                     opp.execution_priority >= 4]
            
            for opportunity in execution_candidates:
                try:
                    # Execute the trade
                    order_result = await self.order_manager.sell_cash_secured_put(
                        symbol=opportunity.symbol,
                        strike=opportunity.put_strike,
                        expiry=opportunity.put_expiry,
                        contracts=opportunity.contracts_to_sell,
                        limit_price=opportunity.premium_target
                    )
                    
                    if order_result['success']:
                        executed_count += 1
                        self.executed_trades.append(opportunity)
                        self.performance.trades_executed += 1
                        
                        logger.info(f"TRADE EXECUTED: {opportunity.symbol} "
                                  f"{opportunity.contracts_to_sell}x {opportunity.put_strike}P "
                                  f"@ ${opportunity.premium_target:.2f}")
                        
                        # Update portfolio state
                        await self.risk_manager.record_new_position(opportunity)
                    
                except Exception as e:
                    logger.error(f"Error executing trade for {opportunity.symbol}: {e}")
                    continue
            
            return executed_count
            
        except Exception as e:
            logger.error(f"Error executing qualified trades: {e}")
            return executed_count
    
    async def _send_opportunity_notifications(self, opportunities: List[TradeOpportunity]):
        """Send notifications for high-priority opportunities"""
        try:
            high_priority_ops = [opp for opp in opportunities if opp.execution_priority >= 4]
            
            if high_priority_ops:
                await self.notification_manager.send_opportunity_alert(high_priority_ops)
                
        except Exception as e:
            logger.debug(f"Error sending notifications: {e}")
    
    async def monitor_existing_positions(self) -> Dict[str, Any]:
        """
        Monitor existing put positions for management opportunities.
        
        Returns:
            Dict with position monitoring results
        """
        try:
            # Get current positions from risk manager
            current_positions = await self.risk_manager.get_current_positions()
            
            monitoring_results = {
                'positions_monitored': len(current_positions),
                'close_recommendations': [],
                'roll_recommendations': [],
                'assignment_alerts': []
            }
            
            for position in current_positions:
                # Check for early close opportunities (50% profit rule)
                if position.unrealized_pnl >= position.max_profit * 0.5:
                    monitoring_results['close_recommendations'].append({
                        'symbol': position.symbol,
                        'reason': '50% profit target reached',
                        'current_profit': position.unrealized_pnl
                    })
                
                # Check for rolling opportunities (21 DTE rule)
                if position.days_to_expiration <= 21:
                    monitoring_results['roll_recommendations'].append({
                        'symbol': position.symbol,
                        'reason': 'Approaching expiration',
                        'dte': position.days_to_expiration
                    })
                
                # Check for assignment risk
                if position.probability_assignment > 0.3:  # >30% assignment risk
                    monitoring_results['assignment_alerts'].append({
                        'symbol': position.symbol,
                        'assignment_probability': position.probability_assignment,
                        'intrinsic_value': position.intrinsic_value
                    })
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return {'error': str(e)}
    
    async def get_strategy_performance(self) -> StrategyPerformance:
        """Get current strategy performance metrics"""
        try:
            # Update performance metrics
            if self.performance.trades_executed > 0:
                self.performance.win_rate = self.performance.trades_profitable / self.performance.trades_executed
            
            # Get current portfolio allocation
            self.performance.current_open_positions = len(await self.risk_manager.get_current_positions())
            self.performance.portfolio_allocation = await self.risk_manager.get_options_allocation()
            
            return self.performance
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return self.performance
    
    def export_opportunities_report(self, opportunities: List[TradeOpportunity], 
                                   filename: str = None) -> str:
        """Export opportunities to detailed report"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"holy_grail_opportunities_{timestamp}.csv"
            
            # Prepare export data
            export_data = []
            for opp in opportunities:
                export_data.append({
                    'Symbol': opp.symbol,
                    'Company': opp.company_name,
                    'Action': opp.recommended_action,
                    'Priority': opp.execution_priority,
                    'Current Price': f"${opp.perfect_storm_signal.current_price:.2f}",
                    'Intrinsic Value': f"${opp.perfect_storm_signal.intrinsic_value:.2f}",
                    'Put Strike': f"${opp.put_strike:.0f}",
                    'Put Expiry': opp.put_expiry,
                    'Contracts': opp.contracts_to_sell,
                    'Premium Target': f"${opp.premium_target:.2f}",
                    'Capital Required': f"${opp.capital_required:,.0f}",
                    'Max Loss': f"${opp.max_loss:,.0f}",
                    'Probability Profit': f"{opp.probability_profit:.1%}",
                    'Annualized Return': f"{opp.annualized_return:.1%}",
                    'Portfolio Impact': f"{opp.portfolio_impact:.1%}",
                    'Entry Score': f"{opp.perfect_storm_signal.entry_score}/100",
                    'Trade Confidence': opp.options_analysis.trade_confidence,
                    'Time Sensitivity': opp.time_sensitivity,
                    'Warnings': '; '.join(opp.trade_warnings) if opp.trade_warnings else 'None'
                })
            
            # Export to CSV
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(opportunities)} opportunities to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting opportunities report: {e}")
            return ""


# Convenience functions for external use
async def run_holy_grail_scan(fmp_api_key: str, portfolio_value: float = 1000000, 
                             execute_trades: bool = False) -> List[TradeOpportunity]:
    """
    Run complete Holy Grail strategy scan.
    
    Args:
        fmp_api_key: Financial Modeling Prep API key
        portfolio_value: Total portfolio value
        execute_trades: Whether to execute qualifying trades
        
    Returns:
        List of trade opportunities
    """
    engine = HolyGrailEntryEngine(fmp_api_key, portfolio_value)
    
    # Initialize strategy
    success = await engine.initialize_strategy()
    if not success:
        logger.error("Failed to initialize strategy")
        return []
    
    # Run full scan
    opportunities = await engine.run_full_scan(execute_trades)
    
    return opportunities


async def quick_opportunity_check(symbols: List[str], fmp_api_key: str, 
                                 portfolio_value: float = 1000000) -> List[TradeOpportunity]:
    """
    Quick check for opportunities in specific symbols.
    
    Args:
        symbols: List of symbols to check
        fmp_api_key: FMP API key
        portfolio_value: Portfolio value
        
    Returns:
        List of opportunities found
    """
    engine = HolyGrailEntryEngine(fmp_api_key, portfolio_value)
    
    # Initialize with custom symbols
    success = await engine.initialize_strategy(symbols)
    if not success:
        return []
    
    # Run scan
    opportunities = await engine.run_full_scan(execute_trades=False)
    
    return opportunities