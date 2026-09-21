# -*- coding: utf-8 -*-
"""The fixed texts every fingerprint is built from, and why these ones.

A drift check is only a comparison if both sides answer the same questions. So
the probe set is frozen here, hashed into every fingerprint, and a check against
a fingerprint built from a different set refuses to run rather than reporting a
number that means nothing.

The set is deliberately spread across the things an embedding model is asked to
do in production, because a model revision rarely degrades everywhere at once:

  * short factual sentences, the ordinary case
  * two near-paraphrases, which must stay close to each other
  * two sentences about different topics, which must stay apart
  * code, which some revisions tokenise differently
  * a non-English sentence, usually the first thing to shift
  * a long paragraph, where truncation limits change behaviour
  * punctuation-heavy text and a near-empty string, the edges

Twelve is enough for 66 pairwise distances, which is a stable enough shape to
correlate, and small enough that a check costs twelve requests.
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
"""ASCII unit separator, built with chr() so no control byte sits in this file.
It cannot occur inside a probe, so two different sets cannot hash alike by
gluing differently."""


def probe_hash(probes: tuple[str, ...] = PROBES) -> str:
    """A short hash of the exact probe set, stored with every fingerprint.

    Twelve hex characters: enough that an accidental collision is not a concern,
    short enough to read in a report and compare by eye.
    """
    joined = SEPARATOR.join(probes).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]
