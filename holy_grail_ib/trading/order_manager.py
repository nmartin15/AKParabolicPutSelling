"""
Order Manager for Holy Grail Options Strategy
Handles trade execution through Interactive Brokers API

Executes cash-secured put selling orders with:
- Smart order routing and timing
- Limit order management
- Position tracking and monitoring
- Error handling and retry logic
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, Order
from ib_insync.objects import Trade, OrderStatus

from config.criteria import get_criteria
from data.ib_client import get_ib_client
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types supported"""
    LIMIT = "LMT"
    MARKET = "MKT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class OrderRequest:
    """Order request specification"""
    symbol: str
    action: str  # "SELL", "BUY"
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Options-specific
    strike: Optional[float] = None
    expiry: Optional[str] = None
    right: Optional[str] = None  # "P" for puts, "C" for calls
    
    # Order management
    time_in_force: str = "DAY"  # "DAY", "GTC", "IOC", "FOK"
    good_after_time: Optional[str] = None
    good_till_date: Optional[str] = None
    
    # Strategy context
    strategy_id: str = "holy_grail"
    trade_id: Optional[str] = None
    notes: str = ""


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str]
    fill_price: Optional[float]
    fill_quantity: Optional[int]
    commission: Optional[float]
    
    # Status tracking
    status: OrderStatus
    status_message: str
    submission_time: datetime
    fill_time: Optional[datetime]
    
    # Error information
    error_code: Optional[str]
    error_message: Optional[str]
    
    # Trade reference
    ib_trade: Optional[Any]  # IB Trade object
    
    
@dataclass
class ActiveOrder:
    """Active order tracking"""
    order_id: str
    order_request: OrderRequest
    ib_order: Order
    ib_trade: Trade
    
    # Status
    current_status: OrderStatus
    submission_time: datetime
    last_update_time: datetime
    
    # Fill information
    filled_quantity: int
    average_fill_price: float
    remaining_quantity: int
    
    # Management
    retry_count: int
    cancel_requested: bool


class OrderManager:
    """
    Manages order execution for Holy Grail options strategy.
    
    Handles all aspects of trade execution including:
    - Order placement and management
    - Fill monitoring and reporting
    - Error handling and retries
    - Position reconciliation
    """
    
    def __init__(self, ib_host: str = '127.0.0.1', ib_port: int = 7497):
        """
        Initialize order manager.
        
        Args:
            ib_host: Interactive Brokers host
            ib_port: Interactive Brokers port
        """
        self.ib_host = ib_host
        self.ib_port = ib_port
        
        # Load criteria
        self.options_criteria = get_criteria('options_selection')
        self.risk_criteria = get_criteria('risk_management')
        
        # Order tracking
        self.active_orders: Dict[str, ActiveOrder] = {}
        self.completed_orders: Dict[str, OrderResult] = {}
        self.order_history: List[OrderResult] = []
        
        # Strategy state
        self.total_orders_placed = 0
        self.total_premium_collected = 0.0
        self.total_commissions_paid = 0.0
        
    async def sell_cash_secured_put(self, symbol: str, strike: float, expiry: str,
                                   contracts: int, limit_price: float = None,
                                   trade_id: str = None) -> OrderResult:
        """
        Sell cash-secured put option.
        
        Args:
            symbol: Stock symbol
            strike: Put strike price
            expiry: Expiration date (YYYYMMDD format)
            contracts: Number of contracts to sell
            limit_price: Limit price (use mid-market if None)
            trade_id: Optional trade identifier
            
        Returns:
            OrderResult with execution details
        """
        try:
            logger.info(f"Selling {contracts}x {symbol} {strike}P exp {expiry} @ ${limit_price}")
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action="SELL",
                quantity=contracts,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                strike=strike,
                expiry=expiry,
                right="P",
                trade_id=trade_id,
                notes=f"Cash-secured put sale - Holy Grail strategy"
            )
            
            # Execute the order
            result = await self._execute_order(order_request)
            
            if result.success:
                logger.info(f"PUT SALE SUCCESS: {symbol} {contracts}x {strike}P filled @ ${result.fill_price}")
                self.total_premium_collected += (result.fill_price * contracts * 100)
                
            return result
            
        except Exception as e:
            logger.error(f"Error selling cash-secured put for {symbol}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                fill_quantity=None,
                commission=None,
                status=OrderStatus.REJECTED,
                status_message=f"Error: {str(e)}",
                submission_time=datetime.now(),
                fill_time=None,
                error_code="EXECUTION_ERROR",
                error_message=str(e),
                ib_trade=None
            )
    
    async def buy_to_close_put(self, symbol: str, strike: float, expiry: str,
                              contracts: int, limit_price: float = None) -> OrderResult:
        """
        Buy to close put position (take profits).
        
        Args:
            symbol: Stock symbol
            strike: Put strike price
            expiry: Expiration date
            contracts: Number of contracts to buy back
            limit_price: Limit price
            
        Returns:
            OrderResult with execution details
        """
        try:
            logger.info(f"Buying to close {contracts}x {symbol} {strike}P exp {expiry}")
            
            order_request = OrderRequest(
                symbol=symbol,
                action="BUY",
                quantity=contracts,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                strike=strike,
                expiry=expiry,
                right="P",
                notes="Buy to close - profit taking"
            )
            
            result = await self._execute_order(order_request)
            
            if result.success:
                logger.info(f"BUY TO CLOSE SUCCESS: {symbol} {contracts}x {strike}P @ ${result.fill_price}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error buying to close put for {symbol}: {e}")
            return self._create_error_result(str(e))
    
    async def roll_put_option(self, current_symbol: str, current_strike: float, 
                             current_expiry: str, new_strike: float, new_expiry: str,
                             contracts: int, net_credit: float = None) -> Dict[str, OrderResult]:
        """
        Roll put option to new strike/expiration.
        
        Args:
            current_symbol: Stock symbol
            current_strike: Current strike price
            current_expiry: Current expiration
            new_strike: New strike price
            new_expiry: New expiration date
            contracts: Number of contracts
            net_credit: Minimum net credit required
            
        Returns:
            Dict with results for both legs of the roll
        """
        try:
            logger.info(f"Rolling {contracts}x {current_symbol} {current_strike}P to {new_strike}P")
            
            # Get current option prices to calculate roll
            current_put_price = await self._get_option_price(current_symbol, current_strike, current_expiry, "P")
            new_put_price = await self._get_option_price(current_symbol, new_strike, new_expiry, "P")
            
            if not current_put_price or not new_put_price:
                raise ValueError("Could not get option prices for roll")
            
            # Calculate net credit/debit
            net_price = new_put_price - current_put_price
            
            if net_credit and net_price < net_credit:
                raise ValueError(f"Roll would result in net debit of ${abs(net_price):.2f}, minimum credit required: ${net_credit:.2f}")
            
            # Execute closing trade first
            close_result = await self.buy_to_close_put(
                current_symbol, current_strike, current_expiry, contracts, current_put_price
            )
            
            # If close successful, open new position
            if close_result.success:
                open_result = await self.sell_cash_secured_put(
                    current_symbol, new_strike, new_expiry, contracts, new_put_price
                )
                
                return {
                    'close_result': close_result,
                    'open_result': open_result,
                    'roll_success': open_result.success,
                    'net_credit': new_put_price - current_put_price
                }
            else:
                return {
                    'close_result': close_result,
                    'open_result': None,
                    'roll_success': False,
                    'error': 'Failed to close existing position'
                }
                
        except Exception as e:
            logger.error(f"Error rolling put option: {e}")
            return {
                'close_result': None,
                'open_result': None,
                'roll_success': False,
                'error': str(e)
            }
    
    async def _execute_order(self, order_request: OrderRequest) -> OrderResult:
        """
        Execute order through Interactive Brokers.
        
        Args:
            order_request: Order specification
            
        Returns:
            OrderResult with execution details
        """
        try:
            # Get IB client
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            
            # Create contract
            if order_request.strike:  # Options order
                contract = Option(
                    symbol=order_request.symbol,
                    lastTradeDateOrContractMonth=order_request.expiry,
                    strike=order_request.strike,
                    right=order_request.right,
                    exchange='SMART'
                )
            else:  # Stock order
                contract = Stock(
                    symbol=order_request.symbol,
                    exchange='SMART',
                    currency='USD'
                )
            
            # Create order
            if order_request.order_type == OrderType.LIMIT:
                if not order_request.limit_price:
                    # Get current market price if no limit specified
                    order_request.limit_price = await self._get_market_price(contract, order_request.action)
                
                ib_order = LimitOrder(
                    action=order_request.action,
                    totalQuantity=order_request.quantity,
                    lmtPrice=order_request.limit_price
                )
            elif order_request.order_type == OrderType.MARKET:
                ib_order = MarketOrder(
                    action=order_request.action,
                    totalQuantity=order_request.quantity
                )
            else:
                raise ValueError(f"Unsupported order type: {order_request.order_type}")
            
            # Set time in force
            ib_order.tif = order_request.time_in_force
            
            # Place order
            trade = ib_client.ib.placeOrder(contract, ib_order)
            
            # Track active order
            active_order = ActiveOrder(
                order_id=str(trade.order.orderId),
                order_request=order_request,
                ib_order=ib_order,
                ib_trade=trade,
                current_status=OrderStatus.SUBMITTED,
                submission_time=datetime.now(),
                last_update_time=datetime.now(),
                filled_quantity=0,
                average_fill_price=0.0,
                remaining_quantity=order_request.quantity,
                retry_count=0,
                cancel_requested=False
            )
            
            self.active_orders[active_order.order_id] = active_order
            self.total_orders_placed += 1
            
            # Wait for fill or timeout
            result = await self._monitor_order_execution(active_order, timeout_seconds=300)
            
            # Update order history
            self.order_history.append(result)
            if result.order_id:
                self.completed_orders[result.order_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return self._create_error_result(str(e))
    
    async def _monitor_order_execution(self, active_order: ActiveOrder, 
                                     timeout_seconds: int = 300) -> OrderResult:
        """
        Monitor order execution until filled or timeout.
        
        Args:
            active_order: Active order to monitor
            timeout_seconds: Timeout in seconds
            
        Returns:
            OrderResult with final status
        """
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout_seconds:
                # Check order status
                trade = active_order.ib_trade
                order_state = trade.orderStatus
                
                # Update active order
                active_order.last_update_time = datetime.now()
                
                if order_state.status == 'Filled':
                    # Order completely filled
                    active_order.current_status = OrderStatus.FILLED
                    active_order.filled_quantity = int(order_state.filled)
                    active_order.average_fill_price = float(order_state.avgFillPrice)
                    active_order.remaining_quantity = 0
                    
                    # Calculate commission
                    commission = 0.0
                    if hasattr(trade, 'commissionReport') and trade.commissionReport:
                        commission = trade.commissionReport.commission
                    
                    self.total_commissions_paid += commission
                    
                    # Remove from active orders
                    if active_order.order_id in self.active_orders:
                        del self.active_orders[active_order.order_id]
                    
                    return OrderResult(
                        success=True,
                        order_id=active_order.order_id,
                        fill_price=active_order.average_fill_price,
                        fill_quantity=active_order.filled_quantity,
                        commission=commission,
                        status=OrderStatus.FILLED,
                        status_message="Order filled successfully",
                        submission_time=active_order.submission_time,
                        fill_time=datetime.now(),
                        error_code=None,
                        error_message=None,
                        ib_trade=trade
                    )
                
                elif order_state.status in ['Cancelled', 'ApiCancelled']:
                    # Order cancelled
                    active_order.current_status = OrderStatus.CANCELLED
                    
                    if active_order.order_id in self.active_orders:
                        del self.active_orders[active_order.order_id]
                    
                    return OrderResult(
                        success=False,
                        order_id=active_order.order_id,
                        fill_price=None,
                        fill_quantity=None,
                        commission=None,
                        status=OrderStatus.CANCELLED,
                        status_message="Order was cancelled",
                        submission_time=active_order.submission_time,
                        fill_time=None,
                        error_code="ORDER_CANCELLED",
                        error_message="Order cancelled before execution",
                        ib_trade=trade
                    )
                
                elif order_state.status == 'PendingSubmit':
                    # Order still pending
                    active_order.current_status = OrderStatus.PENDING
                    
                elif order_state.status == 'Submitted':
                    # Order submitted and working
                    active_order.current_status = OrderStatus.SUBMITTED
                    
                elif order_state.status == 'PartiallyFilled':
                    # Partial fill
                    active_order.current_status = OrderStatus.PARTIALLY_FILLED
                    active_order.filled_quantity = int(order_state.filled)
                    active_order.average_fill_price = float(order_state.avgFillPrice) if order_state.avgFillPrice else 0
                    active_order.remaining_quantity = active_order.order_request.quantity - active_order.filled_quantity
                
                # Wait before next status check
                await asyncio.sleep(1)
            
            # Timeout reached
            logger.warning(f"Order {active_order.order_id} timed out after {timeout_seconds} seconds")
            
            # Cancel the order
            await self._cancel_order(active_order.order_id)
            
            return OrderResult(
                success=False,
                order_id=active_order.order_id,
                fill_price=None,
                fill_quantity=None,
                commission=None,
                status=OrderStatus.CANCELLED,
                status_message="Order timed out and was cancelled",
                submission_time=active_order.submission_time,
                fill_time=None,
                error_code="ORDER_TIMEOUT",
                error_message=f"Order not filled within {timeout_seconds} seconds",
                ib_trade=active_order.ib_trade
            )
            
        except Exception as e:
            logger.error(f"Error monitoring order execution: {e}")
            return self._create_error_result(str(e), active_order.order_id)
    
    async def _get_market_price(self, contract, action: str) -> float:
        """Get current market price for contract"""
        try:
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            ticker = ib_client.ib.reqMktData(contract)
            
            # Wait for price data
            await asyncio.sleep(1)
            
            if action == "SELL":
                return ticker.bid if ticker.bid and ticker.bid == ticker.bid else ticker.last
            else:
                return ticker.ask if ticker.ask and ticker.ask == ticker.ask else ticker.last
                
        except Exception as e:
            logger.debug(f"Error getting market price: {e}")
            return 0.0
    
    async def _get_option_price(self, symbol: str, strike: float, expiry: str, right: str) -> Optional[float]:
        """Get current option price"""
        try:
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            
            contract = Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange='SMART'
            )
            
            ticker = ib_client.ib.reqMktData(contract)
            await asyncio.sleep(1)
            
            # Return mid price
            if ticker.bid and ticker.ask and ticker.bid == ticker.bid and ticker.ask == ticker.ask:
                return (ticker.bid + ticker.ask) / 2
            elif ticker.last and ticker.last == ticker.last:
                return ticker.last
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error getting option price: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        return await self._cancel_order(order_id)
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Internal order cancellation"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            active_order = self.active_orders[order_id]
            active_order.cancel_requested = True
            
            # Cancel through IB
            ib_client = await get_ib_client(self.ib_host, self.ib_port)
            ib_client.ib.cancelOrder(active_order.ib_order)
            
            logger.info(f"Cancellation requested for order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_active_orders(self) -> List[ActiveOrder]:
        """Get list of currently active orders"""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[OrderResult]:
        """Get order execution history"""
        return self.order_history[-limit:]
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        successful_orders = [order for order in self.order_history if order.success]
        
        return {
            'total_orders_placed': self.total_orders_placed,
            'successful_orders': len(successful_orders),
            'success_rate': len(successful_orders) / max(len(self.order_history), 1),
            'total_premium_collected': self.total_premium_collected,
            'total_commissions_paid': self.total_commissions_paid,
            'net_premium_after_commissions': self.total_premium_collected - self.total_commissions_paid,
            'active_orders_count': len(self.active_orders),
            'average_fill_time': self._calculate_average_fill_time()
        }
    
    def _calculate_average_fill_time(self) -> float:
        """Calculate average time to fill orders"""
        try:
            filled_orders = [order for order in self.order_history 
                           if order.success and order.fill_time]
            
            if not filled_orders:
                return 0.0
            
            fill_times = []
            for order in filled_orders:
                fill_time = (order.fill_time - order.submission_time).total_seconds()
                fill_times.append(fill_time)
            
            return sum(fill_times) / len(fill_times)
            
        except Exception as e:
            logger.debug(f"Error calculating average fill time: {e}")
            return 0.0
    
    def _create_error_result(self, error_message: str, order_id: str = None) -> OrderResult:
        """Create error result for failed orders"""
        return OrderResult(
            success=False,
            order_id=order_id,
            fill_price=None,
            fill_quantity=None,
            commission=None,
            status=OrderStatus.REJECTED,
            status_message=error_message,
            submission_time=datetime.now(),
            fill_time=None,
            error_code="EXECUTION_ERROR",
            error_message=error_message,
            ib_trade=None
        )
    
    async def cleanup_old_orders(self, days_old: int = 7):
        """Clean up old order records"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Remove old completed orders
            old_order_ids = [
                order_id for order_id, result in self.completed_orders.items()
                if result.submission_time < cutoff_date
            ]
            
            for order_id in old_order_ids:
                del self.completed_orders[order_id]
            
            # Trim order history
            self.order_history = [
                order for order in self.order_history
                if order.submission_time >= cutoff_date
            ]
            
            logger.info(f"Cleaned up {len(old_order_ids)} old order records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old orders: {e}")


# Convenience functions
async def quick_put_sale(symbol: str, strike: float, expiry: str, 
                        contracts: int, limit_price: float) -> OrderResult:
    """
    Quick cash-secured put sale.
    
    Args:
        symbol: Stock symbol
        strike: Put strike price
        expiry: Expiration date
        contracts: Number of contracts
        limit_price: Limit price
        
    Returns:
        OrderResult
    """
    order_manager = OrderManager()
    return await order_manager.sell_cash_secured_put(
        symbol, strike, expiry, contracts, limit_price
    )


async def close_profitable_puts(profit_threshold: float = 0.5) -> List[OrderResult]:
    """
    Close puts that have reached profit threshold.
    
    Args:
        profit_threshold: Profit threshold (0.5 = 50%)
        
    Returns:
        List of order results for closes
    """
    order_manager = OrderManager()
    # This would need integration with position tracking
    # to identify profitable positions to close
    return []