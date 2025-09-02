from app.utils import profile_schema
from app.core import CSVLlmAssistant
import pandas as pd

def test_profile_counts():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "y"]})
    s = profile_schema(df)
    assert s["rows"] == 3
    assert s["cols"] == 2
    assert s["columns"]["a"]["nulls"] == 1
    assert s["columns"]["b"]["unique"] == 2

def test_rule_based_answers():
    df = pd.DataFrame({
        "product": ["A", "B", "C", "D"],
        "price": [10, 7, 12, 5.5],
        "quantity": [2, 5, 1, 10],
        "purchase_date": pd.to_datetime(["2025-08-01", "2025-08-02", "2025-08-03", "2025-08-01"])
    })
    bot = CSVLlmAssistant(df)

    assert "Columns:" in bot.answer("what are the columns?")
    assert "Row count:" in bot.answer("how many rows?")

    assert "mean(price) =" in bot.answer("average price")
    assert "sum(quantity) =" in bot.answer("total of quantity")
    assert "max(price) =" in bot.answer("maximum price")

    out = bot.answer("unique product")
    assert "distinct(product) =" in out

    out = bot.answer("describe price")
    assert "describe(price) =" in out