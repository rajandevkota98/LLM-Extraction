"""Normalization turns what the text says into what a database can store.

The rules that matter most here are the ones about *refusing* to normalize:
an ambiguous symbol and a relative date must survive as gaps, not guesses.
"""

from __future__ import annotations

from src.components.normalizer import normalize, parse_lead_time


def _payload(**overrides) -> dict:
    payload = {
        "supplier_name": "Beta Metals",
        "currency": "EUR",
        "items": [
            {
                "sku": None,
                "description": "Copper tubing coils",
                "quantity": 12,
                "unit_price": 73,
                "lead_time_days": None,
            }
        ],
        "quote_expiry": None,
        "shipping_included": None,
        "notes": [],
        "assumptions": [],
        "needs_review": False,
    }
    payload.update(overrides)
    return payload


# --- lead time ------------------------------------------------------------- #


def test_weeks_phrase_becomes_days():
    assert parse_lead_time("Delivery around 3 weeks.").days == 21


def test_number_words_and_months_are_handled():
    assert parse_lead_time("Ships in two weeks.").days == 14
    assert parse_lead_time("Lead time 2 months.").days == 60
    assert parse_lead_time("Lead time 14 days.").days == 14


def test_approximation_is_recorded():
    assert parse_lead_time("Delivery around 3 weeks.").approximate is True
    assert parse_lead_time("Lead time 14 days.").approximate is False


def test_quote_validity_duration_is_not_a_lead_time():
    """'Valid for 30 days' is an expiry, not a delivery promise."""
    assert parse_lead_time("Quote valid for 30 days.") is None


def test_lead_time_fills_a_gap_but_never_overwrites():
    result, notes = normalize(_payload(), "Copper coils. Delivery around 3 weeks.")
    assert result["items"][0]["lead_time_days"] == 21
    assert any("21 days derived" in n for n in notes)

    stated = _payload(items=[{**_payload()["items"][0], "lead_time_days": 10}])
    result, _ = normalize(stated, "Copper coils. Delivery around 3 weeks.")
    assert result["items"][0]["lead_time_days"] == 10


# --- currency -------------------------------------------------------------- #


def test_currency_code_is_uppercased():
    result, _ = normalize(_payload(currency="eur"), "Copper coils at eur 73/unit.")
    assert result["currency"] == "EUR"


def test_bare_dollar_sign_stays_unresolved():
    """`$` is four currencies. Guessing USD is exactly the invention we forbid."""
    result, _ = normalize(_payload(currency="$"), "40 units at $18.50 each.")
    assert result["currency"] == "$"


def test_dollar_sign_resolves_when_the_text_names_the_currency():
    result, notes = normalize(_payload(currency="$"), "Currency USD. 40 units at $18.50 each.")
    assert result["currency"] == "USD"
    assert any("USD" in n for n in notes)


def test_dollar_sign_stays_unresolved_when_two_dollar_currencies_are_named():
    result, _ = normalize(_payload(currency="$"), "Prices in USD; CAD also accepted. $18.50 each.")
    assert result["currency"] == "$"


def test_single_currency_symbols_resolve_directly():
    result, _ = normalize(_payload(currency="£"), "12 coils at £73 each.")
    assert result["currency"] == "GBP"


def test_missing_currency_falls_back_to_a_code_in_the_text():
    result, _ = normalize(_payload(currency=None), "Copper tubing coils - qty 12 - EUR 73/unit.")
    assert result["currency"] == "EUR"


# --- expiry, shipping, tidying --------------------------------------------- #


def test_relative_expiry_is_never_resolved():
    result, notes = normalize(_payload(), "Gamma Plastics. Expires next Friday.")
    assert result["quote_expiry"] is None
    assert any("next Friday" in n for n in notes)


def test_shipping_is_inferred_only_when_the_model_left_it_unset():
    result, _ = normalize(_payload(), "Copper coils. Freight extra.")
    assert result["shipping_included"] is False

    result, _ = normalize(_payload(shipping_included=True), "Copper coils. Freight extra.")
    assert result["shipping_included"] is True

    result, _ = normalize(_payload(), "Copper coils. No mention of carriage.")
    assert result["shipping_included"] is None


def test_whitespace_is_trimmed_and_sku_uppercased():
    payload = _payload(
        supplier_name="  Beta   Metals  ",
        items=[{**_payload()["items"][0], "sku": " av-200 ", "description": "  Copper  coils "}],
    )
    result, _ = normalize(payload, "Copper coils.")
    assert result["supplier_name"] == "Beta Metals"
    assert result["items"][0]["sku"] == "AV-200"
    assert result["items"][0]["description"] == "Copper coils"


def test_supplier_casing_is_left_alone():
    """Title-casing would turn 'ACME GmbH' into 'Acme Gmbh'."""
    result, _ = normalize(_payload(supplier_name="ACME GmbH"), "ACME GmbH quote.")
    assert result["supplier_name"] == "ACME GmbH"


def test_blank_and_duplicate_notes_are_dropped():
    result, _ = normalize(_payload(notes=["  ", "Urgent", "Urgent", ""]), "Urgent order.")
    assert result["notes"] == ["Urgent"]
