# test_screening.py
from earnings_calendar_bot import analyze_stock_metrics
import json

# Test with KHC
results = analyze_stock_metrics("KHC")
print(json.dumps(results, indent=2))

# Add other symbols that should pass
symbols = ["KHC", "AAPL", "MSFT", "AMZN"]
for symbol in symbols:
    print(f"\nTesting {symbol}:")
    results = analyze_stock_metrics(symbol)
    if results:
        print(f"  All criteria met: {results['all_criteria_met']}")
        print(f"  Volume: {results['avg_volume']:.0f} - {'PASS' if results['avg_volume_pass'] else 'FAIL'}")
        print(f"  IV/RV: {results['iv30_rv30']:.2f} - {'PASS' if results['iv30_rv30_pass'] else 'FAIL'}")
        print(f"  Slope: {results['ts_slope_0_45']:.6f} - {'PASS' if results['ts_slope_pass'] else 'FAIL'}")