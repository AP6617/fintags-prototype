from app.value_extractor import extract_value_fields
from app.trend_detector import detect

tests = [
    "Revenue increased 12% YoY to $3.2bn.",
    "Net income decreased to $120m in FY2024.",
    "EPS was $0.42 in Q2 2025, up 5% QoQ.",
    "Operating margin fell by 200 bps.",
    "Free cash flow improved to $250m.",
]

for t in tests:
    v = extract_value_fields(t)
    tr = detect(t)
    print("SENT:", t)
    print("  values:", v)
    print("  trend :", tr)
    print("-"*60)
