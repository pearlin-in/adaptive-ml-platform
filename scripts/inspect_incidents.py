import sqlite3
from datetime import datetime

conn = sqlite3.connect("serving/predictions.db")
print("Incidents by model/action:")
for model_id, action, n, first, last in conn.execute(
    "SELECT model_id, action, COUNT(*), MIN(timestamp), MAX(timestamp) FROM incidents GROUP BY model_id, action"
):
    f = datetime.fromtimestamp(first).isoformat(timespec="seconds")
    l = datetime.fromtimestamp(last).isoformat(timespec="seconds")
    print(f"  {model_id:<10} {action:<20} x{n:<6} first {f}  last {l}")

print("\nFirst auto_rollback per model:")
for row in conn.execute(
    "SELECT model_id, version, reason, previous_stable, MIN(timestamp) "
    "FROM incidents WHERE action='auto_rollback' GROUP BY model_id"
):
    print(" ", row)