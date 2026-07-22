"""Review rules decide whether a human has to look at a quote.

The load-bearing property: these rules are ours. The model's opinion is an input,
never the verdict.
"""

from __future__ import annotations

from src.components.reviewer import decide, needs_review


def _clean(**overrides) -> dict:
    payload = {
        "supplier_name": "Alpha Industrial Supplies",
        "currency": "USD",
        "items": [
            {
                "sku": "AV-200",
                "description": "Stainless Valve 2in",
                "quantity": 40,
                "unit_price": 18.5,
                "lead_time_days": 14,
            }
        ],
        "quote_expiry": "2026-08-15",
        "shipping_included": True,
        "notes": [],
        "assumptions": [],
    }
    payload.update(overrides)
    return payload


TEXT = "Alpha Industrial Supplies. Quote valid until 2026-08-15. 40 units at $18.50 each."


def test_a_complete_quote_needs_no_review():
    assert decide(_clean(), TEXT, []) == []
    assert needs_review([], []) is False


def test_missing_supplier_is_flagged():
    assert any("Supplier name" in r for r in decide(_clean(supplier_name=None), TEXT, []))
    assert any("Supplier name" in r for r in decide(_clean(supplier_name="   "), TEXT, []))


def test_unresolved_currency_symbol_is_flagged_as_ambiguous():
    reasons = decide(_clean(currency="$"), TEXT, [])
    assert any("ambiguous" in r for r in reasons)


def test_missing_currency_is_flagged():
    assert any("Currency is missing" in r for r in decide(_clean(currency=None), TEXT, []))


def test_incomplete_items_are_flagged_per_line():
    payload = _clean(
        items=[
            {"sku": None, "description": "", "quantity": 40, "unit_price": 18.5},
            {"sku": None, "description": "Widget", "quantity": 0, "unit_price": None},
        ]
    )
    reasons = decide(payload, TEXT, [])
    assert any("Item 1 is missing a description" in r for r in reasons)
    assert any("Item 2 is missing a usable quantity" in r for r in reasons)
    assert any("Item 2 is missing a usable unit price" in r for r in reasons)


def test_empty_items_is_flagged():
    assert any("No line items" in r for r in decide(_clean(items=[]), TEXT, []))


def test_relative_expiry_is_flagged():
    text = "Gamma Plastics quotation: 100 caps @ 2.2 AED each. Expires next Friday."
    assert any("relative" in r for r in decide(_clean(quote_expiry=None), text, []))


def test_absent_expiry_is_not_flagged():
    """No expiry mentioned at all is fine; an unresolvable one is not."""
    text = "Beta Metals offers copper coils - qty 12 - EUR 73/unit."
    assert decide(_clean(quote_expiry=None), text, []) == []


def test_validation_errors_force_review():
    reasons = decide(_clean(), TEXT, ["items[0].quantity: must be greater than 0, got 0."])
    assert any("schema check" in r for r in reasons)
    assert needs_review([], ["anything"]) is True


def test_model_flag_can_escalate_but_never_clears():
    """The model's own needs_review is a signal, not the decision."""
    assert any("model flagged" in r for r in decide(_clean(), TEXT, [], model_flagged=True))

    # The model saying "fine" cannot override a rule that failed.
    reasons = decide(_clean(supplier_name=None), TEXT, [], model_flagged=False)
    assert any("Supplier name" in r for r in reasons)
    assert needs_review(reasons, []) is True
