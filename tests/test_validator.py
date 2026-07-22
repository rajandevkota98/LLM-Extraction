"""Schema validation is the gate between model output and stored data.

These tests pin the cases where a naive check would let bad data through.
"""

from __future__ import annotations

from src.components.validator import validate_payload


def _payload(**overrides) -> dict:
    """A well-formed payload, so each test changes exactly one thing."""
    payload = {
        "supplier_name": "Alpha Industrial Supplies",
        "currency": "USD",
        "items": [
            {
                "sku": None,
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
        "needs_review": False,
    }
    payload.update(overrides)
    return payload


def _item(**overrides) -> dict:
    return _payload(items=[{**_payload()["items"][0], **overrides}])


def test_well_formed_payload_has_no_errors():
    assert validate_payload(_payload()) == []


def test_null_sku_is_allowed():
    """A missing SKU is normal; only inventing one would be a problem."""
    assert validate_payload(_item(sku=None)) == []


def test_quantity_must_be_a_positive_integer():
    assert any("quantity" in e for e in validate_payload(_item(quantity=0)))
    assert any("quantity" in e for e in validate_payload(_item(quantity=-3)))
    assert any("quantity" in e for e in validate_payload(_item(quantity="12")))
    assert any("quantity" in e for e in validate_payload(_item(quantity=2.5)))


def test_boolean_is_not_a_quantity():
    """bool subclasses int in Python, so `True` passes a naive isinstance check."""
    errors = validate_payload(_item(quantity=True))
    assert any("boolean" in e for e in errors)


def test_integral_float_quantity_is_accepted():
    """JSON has no int/float distinction; 40.0 is a legitimate 40."""
    assert validate_payload(_item(quantity=40.0)) == []


def test_unit_price_must_be_non_negative_and_finite():
    assert any("unit_price" in e for e in validate_payload(_item(unit_price=-1)))
    assert any("unit_price" in e for e in validate_payload(_item(unit_price=float("nan"))))
    assert any("unit_price" in e for e in validate_payload(_item(unit_price=float("inf"))))
    assert validate_payload(_item(unit_price=0)) == []


def test_items_must_be_a_non_empty_list():
    assert any("items" in e for e in validate_payload(_payload(items=[])))
    assert any("items" in e for e in validate_payload(_payload(items="none")))


def test_expiry_must_be_a_real_calendar_date():
    """A regex alone would accept the 31st of February."""
    assert validate_payload(_payload(quote_expiry=None)) == []
    assert any("calendar date" in e for e in validate_payload(_payload(quote_expiry="2026-02-31")))
    assert any("ISO" in e for e in validate_payload(_payload(quote_expiry="15/08/2026")))


def test_notes_and_assumptions_must_be_string_arrays():
    assert any("notes[0]" in e for e in validate_payload(_payload(notes=[1])))
    assert any("assumptions" in e for e in validate_payload(_payload(assumptions="none")))


def test_missing_top_level_keys_are_reported_individually():
    payload = _payload()
    del payload["currency"]
    del payload["notes"]
    errors = validate_payload(payload)
    assert any(e.startswith("currency: required key") for e in errors)
    assert any(e.startswith("notes: required key") for e in errors)


def test_all_problems_are_collected_not_just_the_first():
    """The summary must show every fault, not stop at one."""
    errors = validate_payload(_payload(currency="dollars", items=[], shipping_included="yes"))
    assert len(errors) >= 3


def test_non_object_payload_is_rejected_without_raising():
    assert validate_payload(["not", "an", "object"])
    assert validate_payload(None)
