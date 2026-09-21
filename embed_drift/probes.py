# -*- coding: utf-8 -*-
"""The frozen probe set. A fingerprint stores its hash and refuses to compare
against a different set, because that comparison would mean nothing.

Spread across what a model revision tends to break unevenly: two near-paraphrases
that must stay close, two unrelated sentences that must stay apart, code, SQL,
German, Japanese, a long paragraph near the truncation limit, punctuation-only
text, a bare number and a single space. Twelve gives 66 pairwise distances.
"""

from __future__ import annotations

import hashlib

PROBES: tuple[str, ...] = (
    "The invoice was paid in full on the fourth of March.",
    "Payment for the invoice was completed on 4 March.",
    "A gasket seals the joint between two flanges.",
    "Rainfall in the eastern provinces exceeded the seasonal average.",
    "def refund(order_id: str) -> Decimal:\n    return order_total(order_id) * Decimal('-1')",
    "SELECT customer_id, SUM(amount) FROM payments GROUP BY customer_id HAVING SUM(amount) > 1000",
    "Die Lieferung verzögert sich um zwei Werktage.",
    "配送は二営業日ほど遅れます。",
    (
        "When a vector index is built with one embedding model and queried with another, "
        "nothing raises an error. The nearest neighbours returned are simply the wrong "
        "ones, and the only symptom is that answers get subtly worse over time, which is "
        "usually blamed on the language model rather than on the retrieval step feeding it."
    ),
    "!!! ??? --- *** ... ###",
    "42",
    " ",
)

SEPARATOR = chr(31)
"""Unit separator, built with chr() so no control byte sits in this file. It
cannot occur inside a probe, so two different sets cannot hash alike."""


def probe_hash(probes: tuple[str, ...] = PROBES) -> str:
    joined = SEPARATOR.join(probes).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]
