def convert_currency(amount: float, from_currency: str, to_currency: str, rates: dict) -> float:
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency not in rates or to_currency not in rates:
        raise KeyError("Currency code not found in rates.")
    return round(amount * rates[to_currency] / rates[from_currency], 2)
