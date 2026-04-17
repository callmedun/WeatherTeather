import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio_manager import portfolio_manager, RiskSetting

session = portfolio_manager.Session()
try:
    tp = session.query(RiskSetting).filter_by(key="tp_edge").first()
    if tp:
        tp.value = 0.0
    else:
        session.add(RiskSetting(key="tp_edge", value=0.0))
        
    session.commit()
    print("Database tp_edge updated successfully.")
except Exception as e:
    print("Error:", e)
    session.rollback()
finally:
    session.close()
