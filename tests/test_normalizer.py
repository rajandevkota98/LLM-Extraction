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


def test_payment_terms_and_warranties_are_not_lead_times():
    """Net-30 terms appear on a large share of real quotes.

    Reading one as a delivery promise is silently wrong: an integer lead time
    passes validation and review, so nothing downstream would catch it.
    """
    assert parse_lead_time("Payment terms Net 30 days.") is None
    assert parse_lead_time("Terms: 30 days net.") is None
    assert parse_lead_time("Warranty 12 months on all parts.") is None
    assert parse_lead_time("Invoiced within 14 days.") is None


def test_a_trailing_payment_term_does_not_suppress_a_real_lead_time():
    """Only the text before a duration is context; 'ships in X, net 30' is common."""
    assert parse_lead_time("Ships in 3 weeks, payment net 30.").days == 21


def test_lead_time_fills_a_gap_but_never_overwrites():
    result, notes = normalize(_payload(), "Copper coils. Delivery around 3 weeks.")
    assert result["items"][0]["lead_time_days"] == 21
    assert any("21 days derived" in n for n in notes)

    stated = _payload(items=[{**_payload()["items"][0], "lead_time_days": 10}])
    result, _ = normalize(stated, "Copper coils. Delivery around 3 weeks.")
    assert result["items"][0]["lead_time_days"] == 10


def test_one_lead_time_is_not_spread_across_several_items():
    """The text says '3 weeks' once and never says which line it applies to.

    Copying it onto both would invent an association, and the result passes
    validation, so the invention would ship as a clean record.
    """
    item = _payload()["items"][0]
    two_items = _payload(items=[dict(item), {**item, "description": "Brass fittings"}])
    result, notes = normalize(
        two_items, "Copper coils ship in 3 weeks. Fittings are made to order."
    )

    assert [i["lead_time_days"] for i in result["items"]] == [None, None]
    assert any("does not say which it applies to" in n for n in notes)


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

    result, _ = normalize(_payload(currency=None), "Copper coils at 73 EUR each.")
    assert result["currency"] == "EUR"

    result, _ = normalize(_payload(currency=None), "Beta Metals. Prices in EUR.")
    assert result["currency"] == "EUR"


def test_an_uppercase_word_that_is_not_used_as_a_currency_is_not_one():
    """'ACME SAR OFFICE' is an address. SAR is a real code, which is the trap."""
    result, _ = normalize(_payload(currency=None), "ACME SAR OFFICE. Prices in euros.")
    assert result["currency"] is None


def test_an_unresolvable_model_currency_is_kept_not_replaced():
    """Swapping in a code found elsewhere asserts something the model did not."""
    result, _ = normalize(_payload(currency="Euros"), "Beta GmbH. Wire to our USD account.")
    assert result["currency"] == "Euros"


# --- numbers --------------------------------------------------------------- #


def test_decimal_and_thousands_separators_are_resolved_without_guessing_a_locale():
    """The worst failure available: a finite, valid, silently 1000x-wrong price.

    Where both separators appear the last one is the decimal point, which settles
    both conventions without knowing the supplier's country.
    """
    item = _payload()["items"][0]

    def price(raw):
        result, _ = normalize(_payload(items=[{**item, "unit_price": raw}]), "Copper coils.")
        return result["items"][0]["unit_price"]

    assert price("1.234,56") == 1234.56
    assert price("1,234.56") == 1234.56
    assert price("18,50") == 18.5
    assert price("1,200") == 1200.0
    assert price("12,345,678") == 12345678.0
    assert price("45.00") == 45.0
    assert price("EUR 73.50/unit") == 73.5


def test_a_genuinely_ambiguous_number_survives_as_a_string():
    """Left for the validator to report rather than coerced into a wrong number."""
    item = _payload()["items"][0]

    def price(raw):
        result, _ = normalize(_payload(items=[{**item, "unit_price": raw}]), "Copper coils.")
        return result["items"][0]["unit_price"]

    # Two numbers: which one is the price?
    assert price("12 units @ 5.00") == "12 units @ 5.00"
    assert price("1,23456") == "1,23456"
    assert price("18.50-20.00") == "18.50-20.00"


# --- expiry, shipping, tidying --------------------------------------------- #


def test_relative_expiry_is_never_resolved():
    result, notes = normalize(_payload(), "Gamma Plastics. Expires next Friday.")
    assert result["quote_expiry"] is None
    assert any("next Friday" in n for n in notes)


def test_an_expiry_that_is_not_a_date_is_dropped_and_named():
    """`quote_expiry` is typed as ISO-or-null; free text in it would be parsed
    as a date by whatever stores the record."""
    result, notes = normalize(_payload(quote_expiry="next Friday"), "Expires next Friday.")
    assert result["quote_expiry"] is None
    assert any("next Friday" in n for n in notes)

    result, notes = normalize(_payload(quote_expiry="sometime in Q3"), "Gamma Plastics quote.")
    assert result["quote_expiry"] is None
    assert any("sometime in Q3" in n for n in notes)


def test_a_real_iso_date_is_kept_and_an_impossible_one_is_not():
    result, _ = normalize(_payload(quote_expiry="2026-09-30"), "Valid to 2026-09-30.")
    assert result["quote_expiry"] == "2026-09-30"

    result, _ = normalize(_payload(quote_expiry="2026-02-31"), "Gamma Plastics quote.")
    assert result["quote_expiry"] is None


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
