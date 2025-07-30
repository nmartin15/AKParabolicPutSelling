"""
Holy Grail Options Strategy - Main Entry Point
Complete implementation of Adam Khoo's "Holy Grail" cash-secured put selling strategy

This is the main execution file that orchestrates the entire workflow:
1. Quality screening of S&P 500 universe
2. DCF intrinsic value calculations  
3. Perfect storm signal detection
4. Options chain analysis
5. Risk management validation
6. Trade execution
7. Position monitoring
8. Performance tracking

Usage:
    python main.py --mode scan                    # Daily opportunity scan
    python main.py --mode trade                   # Scan and auto-execute
    python main.py --mode monitor                 # Monitor existing positions  
    python main.py --mode report                  # Generate performance report
    python main.py --symbols AAPL,MSFT,GOOGL     # Custom symbol list
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.entry_engine import HolyGrailEntryEngine, run_holy_grail_scan, quick_opportunity_check
from alerts.notification import NotificationManager, send_opportunity_alert
from utils.logging_setup import setup_logging, get_logger
from config.criteria import get_criteria

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class HolyGrailMain:
    """
    Main orchestrator for the Holy Grail Options Strategy.
    
    Provides different execution modes:
    - Scan: Find opportunities without executing
    - Trade: Find and automatically execute qualifying trades
    - Monitor: Monitor existing positions for management
    - Report: Generate performance and risk reports
    """
    
    def __init__(self, fmp_api_key: str, portfolio_value: float = 1000000,
                 ib_host: str = '127.0.0.1', ib_port: int = 7497):
        """
        Initialize Holy Grail main orchestrator.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key
            portfolio_value: Total portfolio value for position sizing
            ib_host: Interactive Brokers host
            ib_port: Interactive Brokers port (7497=paper, 7496=live)
        """
        self.fmp_api_key = fmp_api_key
        self.portfolio_value = portfolio_value
        self.ib_host = ib_host
        self.ib_port = ib_port
        
        # Initialize main strategy engine
        self.strategy_engine = HolyGrailEntryEngine(
            fmp_api_key=fmp_api_key,
            portfolio_value=portfolio_value,
            ib_host=ib_host,
            ib_port=ib_port
        )
        
        # Initialize notification manager
        self.notification_manager = NotificationManager()
        
        logger.info(f"Holy Grail Strategy initialized - Portfolio: ${portfolio_value:,.0f}")
        logger.info(f"IB Connection: {ib_host}:{ib_port} ({'Paper Trading' if ib_port == 7497 else 'Live Trading'})")
    
    async def run_daily_scan(self, custom_symbols: Optional[List[str]] = None, 
                            execute_trades: bool = False) -> dict:
        """
        Run daily opportunity scan.
        
        Args:
            custom_symbols: Optional custom symbol list
            execute_trades: Whether to auto-execute qualifying trades
            
        Returns:
            Dict with scan results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING HOLY GRAIL DAILY SCAN")
            logger.info("=" * 60)
            
            start_time = datetime.now()
            
            # Initialize strategy if not already done
            logger.info("Step 1: Initializing strategy...")
            success = await self.strategy_engine.initialize_strategy(custom_symbols)
            
            if not success:
                logger.error("Strategy initialization failed")
                return {
                    'success': False,
                    'error': 'Strategy initialization failed',
                    'opportunities': []
                }
            
            # Run full scan
            logger.info("Step 2: Scanning for perfect storm opportunities...")
            opportunities = await self.strategy_engine.run_full_scan(execute_trades)
            
            # Generate summary
            scan_duration = (datetime.now() - start_time).total_seconds()
            
            summary = {
                'success': True,
                'scan_duration_seconds': scan_duration,
                'total_opportunities': len(opportunities),
                'execute_recommended': len([opp for opp in opportunities if opp.recommended_action == "execute"]),
                'monitor_recommended': len([opp for opp in opportunities if opp.recommended_action == "monitor"]),
                'trades_executed': len([opp for opp in opportunities if execute_trades and opp.recommended_action == "execute"]),
                'opportunities': opportunities
            }
            
            # Log summary
            logger.info("=" * 60)
            logger.info("DAILY SCAN COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Duration: {scan_duration:.1f} seconds")
            logger.info(f"Opportunities Found: {summary['total_opportunities']}")
            logger.info(f"Execute Recommended: {summary['execute_recommended']}")
            logger.info(f"Monitor Recommended: {summary['monitor_recommended']}")
            
            if execute_trades:
                logger.info(f"Trades Executed: {summary['trades_executed']}")
            
            # Send notifications for high-priority opportunities
            if opportunities:
                execute_opportunities = [opp for opp in opportunities if opp.recommended_action == "execute"]
                if execute_opportunities:
                    await send_opportunity_alert(execute_opportunities)
                    logger.info(f"Notification sent for {len(execute_opportunities)} high-priority opportunities")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in daily scan: {e}")
            return {
                'success': False,
                'error': str(e),
                'opportunities': []
            }
    
    async def run_position_monitoring(self) -> dict:
        """
        Monitor existing positions for management opportunities.
        
        Returns:
            Dict with monitoring results
        """
        try:
            logger.info("=" * 60)
            logger.info("MONITORING EXISTING POSITIONS")
            logger.info("=" * 60)
            
            # Monitor positions through strategy engine
            monitoring_results = await self.strategy_engine.monitor_existing_positions()
            
            # Log results
            logger.info(f"Positions Monitored: {monitoring_results['positions_monitored']}")
            logger.info(f"Close Recommendations: {len(monitoring_results['close_recommendations'])}")
            logger.info(f"Roll Recommendations: {len(monitoring_results['roll_recommendations'])}")
            logger.info(f"Assignment Alerts: {len(monitoring_results['assignment_alerts'])}")
            
            # Send alerts for important position updates
            if monitoring_results['close_recommendations']:
                for rec in monitoring_results['close_recommendations']:
                    logger.info(f"CLOSE RECOMMENDATION: {rec['symbol']} - {rec['reason']}")
            
            if monitoring_results['assignment_alerts']:
                for alert in monitoring_results['assignment_alerts']:
                    logger.warning(f"ASSIGNMENT RISK: {alert['symbol']} - {alert['assignment_probability']:.1%} probability")
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return {'error': str(e)}
    
    async def generate_performance_report(self) -> dict:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dict with performance metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("GENERATING PERFORMANCE REPORT")
            logger.info("=" * 60)
            
            # Get strategy performance
            performance = await self.strategy_engine.get_strategy_performance()
            
            # Create detailed report
            report = {
                'report_date': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'strategy_performance': {
                    'total_opportunities_found': performance.total_opportunities_found,
                    'trades_executed': performance.trades_executed,
                    'trades_profitable': performance.trades_profitable,
                    'win_rate': performance.win_rate,
                    'total_premium_collected': performance.total_premium_collected,
                    'current_open_positions': performance.current_open_positions,
                    'portfolio_allocation': performance.portfolio_allocation,
                    'annualized_return': performance.annualized_return,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown
                }
            }
            
            # Log key metrics
            logger.info(f"Total Opportunities: {performance.total_opportunities_found}")
            logger.info(f"Trades Executed: {performance.trades_executed}")
            logger.info(f"Win Rate: {performance.win_rate:.1%}")
            logger.info(f"Premium Collected: ${performance.total_premium_collected:,.0f}")
            logger.info(f"Open Positions: {performance.current_open_positions}")
            logger.info(f"Portfolio Allocation: {performance.portfolio_allocation:.1%}")
            logger.info(f"Annualized Return: {performance.annualized_return:.1%}")
            
            # Send performance notification
            await self.notification_manager.send_performance_report(report['strategy_performance'])
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    async def quick_symbol_check(self, symbols: List[str]) -> dict:
        """
        Quick check for opportunities in specific symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dict with results for each symbol
        """
        try:
            logger.info(f"Quick opportunity check for symbols: {', '.join(symbols)}")
            
            opportunities = await quick_opportunity_check(symbols, self.fmp_api_key, self.portfolio_value)
            
            results = {
                'symbols_checked': symbols,
                'opportunities_found': len(opportunities),
                'opportunities': opportunities
            }
            
            # Log results
            for opp in opportunities:
                logger.info(f"{opp.symbol}: Entry Score {opp.perfect_storm_signal.entry_score}/100 - {opp.recommended_action}")
            
            if not opportunities:
                logger.info("No opportunities found in specified symbols")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in quick symbol check: {e}")
            return {'error': str(e), 'opportunities': []}


async def main():
    """Main entry point with command line argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Holy Grail Options Strategy - Adam Khoo's Cash-Secured Put System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode scan                     # Daily opportunity scan
  python main.py --mode trade --portfolio 500000 # Auto-execute with $500k portfolio
  python main.py --mode monitor                  # Monitor existing positions
  python main.py --symbols AAPL,MSFT,GOOGL      # Check specific symbols
  python main.py --mode report                   # Generate performance report
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['scan', 'trade', 'monitor', 'report'],
        default='scan',
        help='Execution mode (default: scan)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to check (e.g., AAPL,MSFT,GOOGL)'
    )
    
    parser.add_argument(
        '--portfolio',
        type=float,
        default=1000000,
        help='Total portfolio value for position sizing (default: 1000000)'
    )
    
    parser.add_argument(
        '--fmp-key',
        type=str,
        default=os.getenv('FMP_API_KEY'),
        help='Financial Modeling Prep API key (or set FMP_API_KEY env var)'
    )
    
    parser.add_argument(
        '--ib-host',
        type=str,
        default='127.0.0.1',
        help='Interactive Brokers host (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--ib-port',
        type=int,
        default=7497,
        help='Interactive Brokers port (7497=paper, 7496=live, default: 7497)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.fmp_key:
        logger.error("FMP API key required. Set --fmp-key or FMP_API_KEY environment variable")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Parse symbols if provided
    custom_symbols = None
    if args.symbols:
        custom_symbols = [symbol.strip().upper() for symbol in args.symbols.split(',')]
        logger.info(f"Custom symbols specified: {custom_symbols}")
    
    # Initialize main orchestrator
    try:
        holy_grail = HolyGrailMain(
            fmp_api_key=args.fmp_key,
            portfolio_value=args.portfolio,
            ib_host=args.ib_host,
            ib_port=args.ib_port
        )
        
        # Execute based on mode
        if args.mode == 'scan':
            logger.info("Mode: SCAN - Finding opportunities without executing trades")
            result = await holy_grail.run_daily_scan(custom_symbols, execute_trades=False)
            
        elif args.mode == 'trade':
            logger.info("Mode: TRADE - Finding and auto-executing qualifying trades")
            result = await holy_grail.run_daily_scan(custom_symbols, execute_trades=True)
            
        elif args.mode == 'monitor':
            logger.info("Mode: MONITOR - Monitoring existing positions")
            result = await holy_grail.run_position_monitoring()
            
        elif args.mode == 'report':
            logger.info("Mode: REPORT - Generating performance report")
            result = await holy_grail.generate_performance_report()
        
        # Handle custom symbols mode
        if custom_symbols and args.mode in ['scan', 'trade']:
            logger.info("Custom symbols mode - checking specific symbols")
            result = await holy_grail.quick_symbol_check(custom_symbols)
        
        # Print final summary
        if result.get('success', True):
            logger.info("✅ Execution completed successfully")
            
            if 'opportunities' in result and result['opportunities']:
                logger.info(f"📊 Summary: {len(result['opportunities'])} opportunities found")
                
                # Print top opportunities
                top_opportunities = result['opportunities'][:3]
                for i, opp in enumerate(top_opportunities, 1):
                    logger.info(f"  {i}. {opp.symbol} - Score: {opp.perfect_storm_signal.entry_score}/100, Action: {opp.recommended_action}")
            
            if args.mode == 'trade' and 'trades_executed' in result:
                logger.info(f"💰 Trades Executed: {result['trades_executed']}")
                
        else:
            logger.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """Entry point when script is run directly"""
    
    print("=" * 80)
    print("🎯 HOLY GRAIL OPTIONS STRATEGY")
    print("   Adam Khoo's Cash-Secured Put Selling System")
    print("=" * 80)
    print()
    
    try:
        # Run main async function
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 Strategy execution interrupted. Goodbye!")
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)