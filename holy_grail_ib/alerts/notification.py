"""
Notification Manager for Holy Grail Options Strategy
Sends real-time alerts for trading opportunities and execution updates

Supports multiple notification channels:
- Discord webhooks for instant alerts
- Email notifications for detailed reports
- SMS alerts for critical events
- Slack integration for team notifications
- Console/log output for development
"""

import asyncio
import aiohttp
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

from config.criteria import get_criteria
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class NotificationConfig:
    """Notification configuration settings"""
    # Discord settings (primary instant alerts)
    discord_webhook_url: Optional[str] = None
    discord_enabled: bool = False
    
    # Email settings (detailed reports)
    email_host: str = "smtp.gmail.com"
    email_port: int = 587
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = None
    email_enabled: bool = False
    
    # Alert preferences
    perfect_storm_alerts: bool = True
    trade_execution_alerts: bool = True
    risk_alerts: bool = True
    performance_reports: bool = True
    
    # Filtering
    min_signal_score: int = 90
    min_urgency_level: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class AlertMessage:
    """Alert message structure"""
    title: str
    message: str
    urgency: str  # "low", "medium", "high", "critical"
    alert_type: str  # "perfect_storm", "trade_execution", "risk_warning", "performance"
    
    # Rich content
    fields: Dict[str, str] = None
    attachments: List[str] = None
    
    # Metadata
    timestamp: datetime = None
    source: str = "holy_grail_strategy"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.fields is None:
            self.fields = {}


class NotificationManager:
    """
    Manages all notification channels for the Holy Grail strategy.
    
    Sends alerts for:
    - Perfect storm signal detection
    - Trade execution updates
    - Risk management warnings
    - Performance reports
    """
    
    def __init__(self, config: NotificationConfig = None):
        """
        Initialize notification manager.
        
        Args:
            config: Notification configuration (loads from environment if None)
        """
        self.config = config or self._load_config_from_env()
        
        # Message history
        self.sent_messages: List[AlertMessage] = []
        self.last_sent_times: Dict[str, datetime] = {}
        
        # Rate limiting
        self.rate_limits = {
            "perfect_storm": timedelta(minutes=5),    # Max 1 per 5 minutes per symbol
            "trade_execution": timedelta(seconds=30), # Max 1 per 30 seconds
            "risk_warning": timedelta(minutes=15),    # Max 1 per 15 minutes
            "performance": timedelta(hours=1)         # Max 1 per hour
        }
        
    def _load_config_from_env(self) -> NotificationConfig:
        """Load notification config from environment variables"""
        return NotificationConfig(
            # Discord (primary channel)
            discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL'),
            discord_enabled=bool(os.getenv('DISCORD_ENABLED', 'true').lower() == 'true'),
            
            # Email (detailed reports)
            email_host=os.getenv('EMAIL_HOST', 'smtp.gmail.com'),
            email_port=int(os.getenv('EMAIL_PORT', 587)),
            email_user=os.getenv('EMAIL_USER'),
            email_password=os.getenv('EMAIL_PASSWORD'),
            email_from=os.getenv('EMAIL_FROM'),
            email_to=os.getenv('EMAIL_TO', '').split(',') if os.getenv('EMAIL_TO') else [],
            email_enabled=bool(os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'),
            
            # Preferences
            perfect_storm_alerts=bool(os.getenv('PERFECT_STORM_ALERTS', 'true').lower() == 'true'),
            trade_execution_alerts=bool(os.getenv('TRADE_EXECUTION_ALERTS', 'true').lower() == 'true'),
            risk_alerts=bool(os.getenv('RISK_ALERTS', 'true').lower() == 'true'),
            performance_reports=bool(os.getenv('PERFORMANCE_REPORTS', 'true').lower() == 'true'),
            
            min_signal_score=int(os.getenv('MIN_SIGNAL_SCORE', 90)),
            min_urgency_level=os.getenv('MIN_URGENCY_LEVEL', 'medium')
        )
    
    async def send_perfect_storm_alert(self, opportunities: List[Any]) -> bool:
        """
        Send alert for perfect storm opportunities.
        
        Args:
            opportunities: List of TradeOpportunity objects
            
        Returns:
            True if alert sent successfully
        """
        try:
            if not self.config.perfect_storm_alerts or not opportunities:
                return True
            
            # Filter opportunities by score
            high_score_opportunities = [
                opp for opp in opportunities 
                if opp.perfect_storm_signal.entry_score >= self.config.min_signal_score
            ]
            
            if not high_score_opportunities:
                return True
            
            # Check rate limiting
            rate_limit_key = f"perfect_storm_{datetime.now().strftime('%Y%m%d_%H')}"
            if self._is_rate_limited(rate_limit_key, "perfect_storm"):
                logger.debug("Perfect storm alert rate limited")
                return True
            
            # Create alert message
            top_opportunity = max(high_score_opportunities, key=lambda x: x.perfect_storm_signal.entry_score)
            
            alert = AlertMessage(
                title="🎯 PERFECT STORM DETECTED - Holy Grail Strategy",
                message=self._format_perfect_storm_message(high_score_opportunities),
                urgency=self._map_urgency(top_opportunity.urgency_level),
                alert_type="perfect_storm",
                fields=self._create_opportunity_fields(top_opportunity)
            )
            
            # Send through all enabled channels
            success = await self._send_multi_channel_alert(alert)
            
            if success:
                self.last_sent_times[rate_limit_key] = datetime.now()
                logger.info(f"Perfect storm alert sent for {len(high_score_opportunities)} opportunities")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending perfect storm alert: {e}")
            return False
    
    def _format_perfect_storm_message(self, opportunities: List[Any]) -> str:
        """Format perfect storm opportunities message"""
        if len(opportunities) == 1:
            opp = opportunities[0]
            return (
                f"**{opp.symbol} - {opp.company_name}**\n"
                f"📊 Entry Score: {opp.perfect_storm_signal.entry_score}/100\n"
                f"💰 Current: ${opp.current_price:.2f} vs Intrinsic: ${opp.intrinsic_value:.2f}\n"
                f"📉 Drop: {opp.perfect_storm_signal.technical_data.single_day_drop_pct:.1f}% on {opp.perfect_storm_signal.technical_data.volume_spike_ratio:.1f}x volume\n"
                f"📈 VIX: {opp.perfect_storm_signal.current_vix:.1f}\n"
                f"🎯 Put Strike: ${opp.put_strike:.0f} for ${opp.premium_target:.2f} premium\n"
                f"⚡ Urgency: {opp.urgency_level.upper()}\n"
                f"✅ Action: {opp.recommended_action.upper()}"
            )
        else:
            message = f"**{len(opportunities)} Perfect Storm Opportunities Detected:**\n\n"
            for i, opp in enumerate(opportunities[:5], 1):  # Top 5
                message += (
                    f"{i}. **{opp.symbol}** - Score: {opp.perfect_storm_signal.entry_score}/100, "
                    f"Drop: {opp.perfect_storm_signal.technical_data.single_day_drop_pct:.1f}%, "
                    f"Action: {opp.recommended_action}\n"
                )
            
            if len(opportunities) > 5:
                message += f"\n...and {len(opportunities) - 5} more opportunities"
            
            return message
    
    def _create_opportunity_fields(self, opportunity: Any) -> Dict[str, str]:
        """Create rich fields for opportunity alert"""
        return {
            "Symbol": opportunity.symbol,
            "Entry Score": f"{opportunity.perfect_storm_signal.entry_score}/100",
            "Current Price": f"${opportunity.current_price:.2f}",
            "Intrinsic Value": f"${opportunity.intrinsic_value:.2f}",
            "Margin of Safety": f"{opportunity.perfect_storm_signal.margin_of_safety:.1%}",
            "Put Strike": f"${opportunity.put_strike:.0f}",
            "Premium Target": f"${opportunity.premium_target:.2f}",
            "Contracts": str(opportunity.contracts_to_sell),
            "Capital Required": f"${opportunity.capital_required:,.0f}",
            "Ann. Return": f"{opportunity.annualized_return:.1%}",
            "VIX Level": f"{opportunity.perfect_storm_signal.current_vix:.1f}",
            "Urgency": opportunity.urgency_level.title(),
            "Confidence": opportunity.options_analysis.trade_confidence.title()
        }
    
    async def send_trade_execution_alert(self, trade_result: Dict[str, Any]) -> bool:
        """
        Send alert for trade execution.
        
        Args:
            trade_result: Trade execution result
            
        Returns:
            True if alert sent successfully
        """
        try:
            if not self.config.trade_execution_alerts:
                return True
            
            # Check rate limiting
            rate_limit_key = f"trade_execution_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if self._is_rate_limited(rate_limit_key, "trade_execution"):
                return True
            
            # Determine alert urgency based on result
            if trade_result.get('success'):
                urgency = "medium"
                title = "✅ TRADE EXECUTED - Holy Grail Strategy"
                emoji = "✅"
            else:
                urgency = "high"
                title = "❌ TRADE FAILED - Holy Grail Strategy"
                emoji = "❌"
            
            message = self._format_trade_execution_message(trade_result, emoji)
            
            alert = AlertMessage(
                title=title,
                message=message,
                urgency=urgency,
                alert_type="trade_execution",
                fields=self._create_trade_execution_fields(trade_result)
            )
            
            success = await self._send_multi_channel_alert(alert)
            
            if success:
                self.last_sent_times[rate_limit_key] = datetime.now()
                logger.info(f"Trade execution alert sent: {trade_result.get('symbol', 'Unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending trade execution alert: {e}")
            return False
    
    def _format_trade_execution_message(self, trade_result: Dict[str, Any], emoji: str) -> str:
        """Format trade execution message"""
        symbol = trade_result.get('symbol', 'Unknown')
        
        if trade_result.get('success'):
            return (
                f"{emoji} **PUT SOLD SUCCESSFULLY**\n"
                f"**{symbol}** - {trade_result.get('contracts', 0)}x ${trade_result.get('strike', 0):.0f}P\n"
                f"💰 Premium: ${trade_result.get('fill_price', 0):.2f} per contract\n"
                f"📅 Expiry: {trade_result.get('expiry', 'Unknown')}\n"
                f"💵 Total Premium: ${trade_result.get('total_premium', 0):,.0f}\n"
                f"🎯 Strategy: Cash-Secured Put Sale"
            )
        else:
            return (
                f"{emoji} **TRADE EXECUTION FAILED**\n"
                f"**{symbol}** - {trade_result.get('contracts', 0)}x ${trade_result.get('strike', 0):.0f}P\n"
                f"❌ Error: {trade_result.get('error_message', 'Unknown error')}\n"
                f"🔄 Retry recommended: {trade_result.get('retry_recommended', 'No')}"
            )
    
    def _create_trade_execution_fields(self, trade_result: Dict[str, Any]) -> Dict[str, str]:
        """Create fields for trade execution alert"""
        fields = {
            "Symbol": trade_result.get('symbol', 'Unknown'),
            "Action": "SELL PUT",
            "Contracts": str(trade_result.get('contracts', 0)),
            "Strike": f"${trade_result.get('strike', 0):.0f}",
            "Expiry": trade_result.get('expiry', 'Unknown')
        }
        
        if trade_result.get('success'):
            fields.update({
                "Fill Price": f"${trade_result.get('fill_price', 0):.2f}",
                "Total Premium": f"${trade_result.get('total_premium', 0):,.0f}",
                "Commission": f"${trade_result.get('commission', 0):.2f}",
                "Order ID": str(trade_result.get('order_id', 'Unknown'))
            })
        else:
            fields.update({
                "Error": trade_result.get('error_message', 'Unknown'),
                "Error Code": trade_result.get('error_code', 'Unknown')
            })
        
        return fields
    
    async def send_risk_warning(self, risk_warning: Dict[str, Any]) -> bool:
        """
        Send risk management warning.
        
        Args:
            risk_warning: Risk warning details
            
        Returns:
            True if alert sent successfully
        """
        try:
            if not self.config.risk_alerts:
                return True
            
            # Check rate limiting
            rate_limit_key = f"risk_warning_{risk_warning.get('type', 'general')}"
            if self._is_rate_limited(rate_limit_key, "risk_warning"):
                return True
            
            alert = AlertMessage(
                title="⚠️ RISK WARNING - Holy Grail Strategy",
                message=self._format_risk_warning_message(risk_warning),
                urgency="high",
                alert_type="risk_warning",
                fields=self._create_risk_warning_fields(risk_warning)
            )
            
            success = await self._send_multi_channel_alert(alert)
            
            if success:
                self.last_sent_times[rate_limit_key] = datetime.now()
                logger.info(f"Risk warning sent: {risk_warning.get('type', 'Unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending risk warning: {e}")
            return False
    
    def _format_risk_warning_message(self, risk_warning: Dict[str, Any]) -> str:
        """Format risk warning message"""
        return (
            f"⚠️ **{risk_warning.get('type', 'RISK WARNING').upper()}**\n"
            f"{risk_warning.get('message', 'Risk limit exceeded')}\n\n"
            f"**Current Status:**\n"
            f"• Portfolio Allocation: {risk_warning.get('current_allocation', 0):.1%}\n"
            f"• Limit: {risk_warning.get('limit', 0):.1%}\n"
            f"• Recommended Action: {risk_warning.get('recommendation', 'Review positions')}"
        )
    
    def _create_risk_warning_fields(self, risk_warning: Dict[str, Any]) -> Dict[str, str]:
        """Create fields for risk warning"""
        return {
            "Warning Type": risk_warning.get('type', 'Unknown'),
            "Current Value": f"{risk_warning.get('current_allocation', 0):.1%}",
            "Limit": f"{risk_warning.get('limit', 0):.1%}",
            "Severity": risk_warning.get('severity', 'Medium'),
            "Action Required": risk_warning.get('recommendation', 'Review')
        }
    
    async def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Send performance summary report.
        
        Args:
            performance_data: Performance metrics
            
        Returns:
            True if report sent successfully
        """
        try:
            if not self.config.performance_reports:
                return True
            
            # Check rate limiting
            rate_limit_key = "performance_report"
            if self._is_rate_limited(rate_limit_key, "performance"):
                return True
            
            alert = AlertMessage(
                title="📊 STRATEGY PERFORMANCE REPORT - Holy Grail",
                message=self._format_performance_message(performance_data),
                urgency="low",
                alert_type="performance",
                fields=self._create_performance_fields(performance_data)
            )
            
            success = await self._send_multi_channel_alert(alert)
            
            if success:
                self.last_sent_times[rate_limit_key] = datetime.now()
                logger.info("Performance report sent")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending performance report: {e}")
            return False
    
    def _format_performance_message(self, performance: Dict[str, Any]) -> str:
        """Format performance report message"""
        return (
            f"📊 **HOLY GRAIL STRATEGY PERFORMANCE**\n\n"
            f"🎯 **Trading Results:**\n"
            f"• Total Opportunities: {performance.get('total_opportunities', 0)}\n"
            f"• Trades Executed: {performance.get('trades_executed', 0)}\n"
            f"• Win Rate: {performance.get('win_rate', 0):.1%}\n"
            f"• Premium Collected: ${performance.get('total_premium', 0):,.0f}\n\n"
            f"📈 **Performance Metrics:**\n"
            f"• Annualized Return: {performance.get('annualized_return', 0):.1%}\n"
            f"• Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n"
            f"• Max Drawdown: {performance.get('max_drawdown', 0):.1%}\n"
            f"• Active Positions: {performance.get('active_positions', 0)}"
        )
    
    def _create_performance_fields(self, performance: Dict[str, Any]) -> Dict[str, str]:
        """Create fields for performance report"""
        return {
            "Total Opportunities": str(performance.get('total_opportunities', 0)),
            "Trades Executed": str(performance.get('trades_executed', 0)),
            "Win Rate": f"{performance.get('win_rate', 0):.1%}",
            "Premium Collected": f"${performance.get('total_premium', 0):,.0f}",
            "Annualized Return": f"{performance.get('annualized_return', 0):.1%}",
            "Sharpe Ratio": f"{performance.get('sharpe_ratio', 0):.2f}",
            "Active Positions": str(performance.get('active_positions', 0)),
            "Portfolio Allocation": f"{performance.get('portfolio_allocation', 0):.1%}"
        }
    
    async def _send_multi_channel_alert(self, alert: AlertMessage) -> bool:
        """Send alert through enabled channels"""
        success = True
        
        # Discord (primary instant alerts)
        if self.config.discord_enabled and self.config.discord_webhook_url:
            try:
                await self._send_discord_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send Discord alert: {e}")
                success = False
        
        # Email (detailed reports, especially for performance summaries)
        if self.config.email_enabled and self.config.email_to and alert.alert_type in ["performance", "risk_warning"]:
            try:
                await self._send_email_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
                success = False
        
        # Always log to console
        self._log_alert(alert)
        
        # Store in history
        self.sent_messages.append(alert)
        
        return success
    
    async def _send_discord_alert(self, alert: AlertMessage):
        """Send alert to Discord webhook"""
        try:
            # Create Discord embed
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": self._get_color_for_urgency(alert.urgency),
                "timestamp": alert.timestamp.isoformat(),
                "footer": {
                    "text": f"Holy Grail Strategy • {alert.alert_type}"
                }
            }
            
            # Add fields if available
            if alert.fields:
                embed["fields"] = [
                    {"name": key, "value": value, "inline": True}
                    for key, value in list(alert.fields.items())[:10]  # Discord limit
                ]
            
            payload = {
                "embeds": [embed],
                "username": "Holy Grail Bot"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.discord_webhook_url, json=payload) as response:
                    if response.status == 204:
                        logger.debug("Discord alert sent successfully")
                    else:
                        logger.error(f"Discord webhook failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            raise
    
    def _get_color_for_urgency(self, urgency: str) -> int:
        """Get Discord embed color for urgency level"""
        colors = {
            "low": 0x00ff00,      # Green
            "medium": 0xffff00,   # Yellow  
            "high": 0xff8800,     # Orange
            "critical": 0xff0000  # Red
        }
        return colors.get(urgency, 0x0099ff)  # Default blue
    
    async def _send_email_alert(self, alert: AlertMessage):
        """Send email alert"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = alert.title
            
            # Create HTML body
            html_body = self._create_html_email_body(alert)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.email_host, self.config.email_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_from, self.config.email_to, text)
            server.quit()
            
            logger.debug("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            raise
    
    def _create_html_email_body(self, alert: AlertMessage) -> str:
        """Create HTML email body"""
        html = f"""
        <html>
        <body>
            <h2>{alert.title}</h2>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Urgency:</strong> {alert.urgency.upper()}</p>
            
            <div style="background-color: #f5f5f5; padding: 15px; margin: 10px 0;">
                {alert.message.replace('\n', '<br>')}
            </div>
        """
        
        if alert.fields:
            html += "<h3>Details:</h3><table border='1' style='border-collapse: collapse;'>"
            for key, value in alert.fields.items():
                html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
            html += "</table>"
        
        html += """
            <p><em>Generated by Holy Grail Options Strategy</em></p>
        </body>
        </html>
        """
        
        return html
    
    async def _send_sms_alert(self, alert: AlertMessage):
        """Send SMS alert using Twilio"""
        try:
            from twilio.rest import Client
            
            client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
            
            # Create short SMS message
            sms_message = f"{alert.title}\n{alert.message[:100]}..."
            
            for number in self.config.sms_to_numbers:
                message = client.messages.create(
                    body=sms_message,
                    from_=self.config.twilio_from_number,
                    to=number
                )
                logger.debug(f"SMS sent to {number}: {message.sid}")
                
        except Exception as e:
            logger.error(f"Error sending SMS alert: {e}")
            raise
    
    async def _send_slack_alert(self, alert: AlertMessage):
        """Send alert to Slack webhook"""
        try:
            payload = {
                "channel": self.config.slack_channel,
                "username": "Holy Grail Bot",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": "good" if alert.urgency == "low" else "warning" if alert.urgency == "medium" else "danger",
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {"title": key, "value": value, "short": True}
                            for key, value in (alert.fields or {}).items()
                        ][:10],
                        "footer": "Holy Grail Strategy",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.debug("Slack alert sent successfully")
                    else:
                        logger.error(f"Slack webhook failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            raise
    
    def _log_alert(self, alert: AlertMessage):
        """Log alert to console/file"""
        log_message = f"[{alert.urgency.upper()}] {alert.title}: {alert.message}"
        
        if alert.urgency == "critical":
            logger.critical(log_message)
        elif alert.urgency == "high":
            logger.error(log_message)
        elif alert.urgency == "medium":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _is_rate_limited(self, key: str, alert_type: str) -> bool:
        """Check if alert is rate limited"""
        if key not in self.last_sent_times:
            return False
        
        time_since_last = datetime.now() - self.last_sent_times[key]
        rate_limit = self.rate_limits.get(alert_type, timedelta(minutes=5))
        
        return time_since_last < rate_limit
    
    def _map_urgency(self, strategy_urgency: str) -> str:
        """Map strategy urgency to notification urgency"""
        mapping = {
            "low": "low",
            "medium": "medium", 
            "high": "high",
            "critical": "critical"
        }
        return mapping.get(strategy_urgency, "medium")
    
    def get_alert_history(self, hours: int = 24) -> List[AlertMessage]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.sent_messages
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert sending statistics"""
        total_alerts = len(self.sent_messages)
        
        if total_alerts == 0:
            return {
                'total_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_urgency': {},
                'success_rate': 0.0
            }
        
        # Count by type
        alerts_by_type = {}
        for alert in self.sent_messages:
            alert_type = alert.alert_type
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Count by urgency
        alerts_by_urgency = {}
        for alert in self.sent_messages:
            urgency = alert.urgency
            alerts_by_urgency[urgency] = alerts_by_urgency.get(urgency, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'alerts_by_type': alerts_by_type,
            'alerts_by_urgency': alerts_by_urgency,
            'success_rate': 1.0,  # Simplified - would track actual success/failure
            'last_24h_count': len(self.get_alert_history(24))
        }


# Convenience functions for external use
async def send_opportunity_alert(opportunities: List[Any]) -> bool:
    """
    Send perfect storm opportunity alert.
    
    Args:
        opportunities: List of TradeOpportunity objects
        
    Returns:
        True if alert sent successfully
    """
    notification_manager = NotificationManager()
    return await notification_manager.send_perfect_storm_alert(opportunities)


async def send_trade_alert(trade_result: Dict[str, Any]) -> bool:
    """
    Send trade execution alert.
    
    Args:
        trade_result: Trade execution result
        
    Returns:
        True if alert sent successfully
    """
    notification_manager = NotificationManager()
    return await notification_manager.send_trade_execution_alert(trade_result)


async def send_risk_alert(risk_warning: Dict[str, Any]) -> bool:
    """
    Send risk management alert.
    
    Args:
        risk_warning: Risk warning details
        
    Returns:
        True if alert sent successfully
    """
    notification_manager = NotificationManager()
    return await notification_manager.send_risk_warning(risk_warning)


# Example usage and testing
async def test_notifications():
    """Test notification system with sample data"""
    
    # Create test configuration
    config = NotificationConfig(
        discord_webhook_url="YOUR_DISCORD_WEBHOOK_URL",
        discord_enabled=True,
        email_enabled=False,  # Set to True to test email
    )
    
    notification_manager = NotificationManager(config)
    
    # Test perfect storm alert
    print("Testing perfect storm alert...")
    sample_opportunities = []  # Would contain TradeOpportunity objects
    
    # Test trade execution alert
    print("Testing trade execution alert...")
    sample_trade_result = {
        'success': True,
        'symbol': 'AAPL',
        'contracts': 2,
        'strike': 150.0,
        'expiry': '20240315',
        'fill_price': 3.50,
        'total_premium': 700.0,
        'commission': 2.00,
        'order_id': 'TEST123'
    }
    
    await notification_manager.send_trade_execution_alert(sample_trade_result)
    
    # Test risk warning
    print("Testing risk warning alert...")
    sample_risk_warning = {
        'type': 'Position Size Limit',
        'message': 'Portfolio allocation approaching maximum limit',
        'current_allocation': 0.22,
        'limit': 0.25,
        'recommendation': 'Consider closing some positions',
        'severity': 'Medium'
    }
    
    await notification_manager.send_risk_warning(sample_risk_warning)
    
    print("Notification tests completed!")


if __name__ == "__main__":
    # Run notification tests
    import asyncio
    asyncio.run(test_notifications())