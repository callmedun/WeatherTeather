import sys
import os

# Add the parent directory to Python path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio_manager import portfolio_manager, RiskSetting

session = portfolio_manager.Session()
try:
    sl = session.query(RiskSetting).filter_by(key="sl_pnl").first()
    if sl:
        sl.value = -70.0
    else:
        session.add(RiskSetting(key="sl_pnl", value=-70.0))
        
    tp = session.query(RiskSetting).filter_by(key="strong_tp_pnl").first()
    if tp:
        tp.value = 60.0
    else:
        session.add(RiskSetting(key="strong_tp_pnl", value=60.0))
        
    session.commit()
    print("Database RiskSettings updated successfully.")
except Exception as e:
    print("Error:", e)
    session.rollback()
finally:
    session.close()
