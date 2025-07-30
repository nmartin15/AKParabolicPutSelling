# Holy Grail Options Strategy

Implementation of Adam Khoo's "Holy Grail" cash-secured put selling strategy using Interactive Brokers and Financial Modeling Prep APIs.

## Strategy Overview

This system implements Adam Khoo's systematic approach to selling cash-secured puts on high-quality companies during "perfect storm" conditions:

1. **Business Quality Screening**: Filter for companies with consistent growth, strong balance sheets, and competitive moats
2. **Intrinsic Value Calculation**: DCF-based fair value assessment
3. **Perfect Storm Detection**: Parabolic drops + high VIX + strong support levels
4. **Options Analysis**: Find optimal put strikes with attractive premiums
5. **Position Management**: Track trades, handle rolling and assignments

## Setup

### Prerequisites
- Python 3.9+
- Interactive Brokers account with TWS/Gateway running
- Financial Modeling Prep API subscription

### Installation
```bash
git clone <repository>
cd holy_grail_ib
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
1. Copy `.env.template` to `.env`
2. Fill in your API keys and account details
3. Ensure TWS/IB Gateway is running on correct port

### First Run
```bash
python main.py --scan-only  # Dry run to test screening
python main.py              # Full execution with alerts
```

## Project Structure

```
holy_grail_ib/
├── config/          # Configuration and criteria definitions
├── data/            # Data providers (IB, FMP)
├── screening/       # Business quality filters and valuation
├── signals/         # Perfect storm detection logic
├── options/         # Options chain analysis
├── trading/         # Order execution and position management
├── strategy/        # High-level strategy orchestration
├── alerts/          # Notification system
├── utils/           # Utilities and helpers
├── main.py          # Daily strategy execution
└── live_monitor.py  # Real-time monitoring
```

## Key Features

- **Real-time Options Data**: Live Greeks, IV, and pricing via Interactive Brokers
- **Fundamental Analysis**: Comprehensive business quality screening
- **Risk Management**: Position sizing, portfolio limits, correlation checks
- **Automated Alerts**: Email/SMS notifications for trading opportunities
- **Paper Trading**: Test strategies without risking capital
- **Position Tracking**: Monitor open trades and rolling opportunities

## Usage

### Daily Screening
```bash
python main.py
```
Runs morning screening routine, updates watchlists, sends alerts for qualified opportunities.

### Live Monitoring
```bash
python live_monitor.py
```
Real-time monitoring for perfect storm conditions during market hours.

### Manual Analysis
```python
from screening.screener import QualityScreener
from signals.perfect_storm import PerfectStormDetector

# Screen for quality companies
screener = QualityScreener()
candidates = screener.scan_market()

# Check for entry signals
detector = PerfectStormDetector()
opportunities = detector.find_signals(candidates)
```

## Risk Warnings

- **Real Money Trading**: This system can place actual trades. Always test thoroughly in paper trading first.
- **Market Risk**: Options trading involves substantial risk. Never risk more than you can afford to lose.
- **System Risk**: Ensure proper error handling and position monitoring.
- **Data Dependencies**: Strategy relies on real-time data feeds. Monitor for outages.

## Configuration

All strategy parameters are defined in `config/criteria.py`:
- Business quality thresholds
- Entry signal requirements  
- Risk management limits
- Options selection criteria

Modify these values to adjust strategy behavior.

## Development

### Testing
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

## License

Private use only. Not for redistribution.

## Disclaimer

This software is for educational and personal use only. Trading involves substantial risk. The authors are not responsible for any trading losses.