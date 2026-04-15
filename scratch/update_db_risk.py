import sqlite3
import os

db_path = os.path.join("data", "portfolio.db")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Update settings
    updates = [
        ("tp_edge", 5.0),
        ("sl_pnl", -30.0)
    ]
    
    for key, val in updates:
        cursor.execute("UPDATE risk_settings SET value = ? WHERE key = ?", (val, key))
        if cursor.rowcount == 0:
            cursor.execute("INSERT INTO risk_settings (key, value) VALUES (?, ?)", (key, val))
    
    conn.commit()
    conn.close()
    print("Database updated successfully with new risk thresholds.")
else:
    print("Database file not found. Defaults will be applied on next start.")
