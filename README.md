# embed-drift

Catches the failure that raises no error: your vector index was built with one
embedding model, and the provider quietly swapped it for another.

```bash
py -m embed_drift.cli baseline --model all-minilm --out fingerprint.json
# weeks later, in CI
py -m embed_drift.cli check fingerprint.json
```

Exit 0 when the space is where you left it, 1 when it moved, 2 when the question
could not be asked at all. Python standard library only, no dependencies. Works
against Ollama on your own machine or any OpenAI-shaped `/v1/embeddings`
endpoint; a key is read from an environment variable you name and is never
printed, never written to a file and never passed on a command line.

## Why this exists

Nothing breaks loudly when an embedding model changes under you. No exception,
no 4xx, no log line. The nearest neighbours your retrieval returns are simply
the wrong ones, answers get subtly worse, and the blame lands on the language
model rather than on the retrieval step feeding it. OpenAI revised `ada-002`
more than once while the name stayed the same, and every write-up of the problem
ends with the same advice — pin the version, keep a fingerprint — without
shipping anything that actually does it.

This does that one thing.

## The interesting part: two numbers, not one

The obvious check is to embed the same text twice and take the cosine between
old and new. On its own it lies in both directions.

It **cries wolf**, because a provider can renormalise or rescale without
changing what the model knows. Every vector moves, every cosine drops, and
retrieval is no worse than before.

It also **stays quiet** when it should not. A mean cosine of 0.95 sounds
reassuring, but if each vector rotated by a slightly *different* amount, the
neighbourhoods your index depends on have already reshuffled.

So `check` reports two things:

| | What it measures | What moves it |
|---|---|---|
| `direct` | cosine between the old and new vector for the same probe | any change at all, harmless ones included |
| `geometry` | correlation between the *pairwise distances among probes*, old vs new | only a real rearrangement of what is near what |

`geometry` is the constellation's shape rather than where it sits. A uniform
rescaling moves every point and leaves every angle intact, so geometry stays at
1. A genuinely different model changes which probes are near which, and geometry
falls. That is the number that predicts whether your index still retrieves the
right chunk.

Read together:

- **direct low, geometry high** — the encoding changed, the knowledge did not.
  Reindex, and retrieval quality is probably intact once you do.
- **both low** — this is the bad one. Reindex, then *re-measure* retrieval
  quality rather than assuming it came back.
- **dimension changed** — nothing old is comparable to anything new. Hard fail,
  no further arithmetic attempted.

The synthetic proof is in the test suite: a uniform rotation of every vector
drops `direct` below 0.95 while `geometry` stays at 1.0 to six decimal places.
Without that split, the tool would report a model change every time a provider
touched its normalisation.

## A real run

Baseline taken with `all-minilm` (384 dimensions, probe-set hash `867a29c5b7e3`)
on 21 September 2026, then checked three ways on the same machine:

| Checked against | direct (mean) | geometry | Verdict | Exit |
|---|---|---|---|---|
| `all-minilm` — the same model | 1.0000 | 1.0000 | the space is where you left it | 0 |
| `snowflake-arctic-embed:s` — different model, same 384 dimensions | **-0.0052** | **0.3494** | the shape changed | 1 |
| `nomic-embed-text` — 768 dimensions | — | — | dimension changed, nothing comparable | 1 |

Two things worth reading off that middle row.

A `direct` mean of **-0.0052** means the two models' vectors are effectively
orthogonal: different model families do not share a coordinate system at all, so
"how far did my vector move" saturates immediately and stops being informative.
That is precisely why the second number exists.

`geometry` at **0.349** is the more interesting figure. It is far below the 0.99
floor — the neighbourhoods have genuinely been rearranged — but it is not zero
either, because both models do agree that two paraphrases belong closer together
than two unrelated sentences. The partial agreement is real, and it is not
enough: an index built on one and queried with the other returns the wrong rows.

The probe that moved most was #9, the punctuation-only string. Symbols with no
semantic content are where embedding models differ most freely, which makes that
probe an early warning rather than a curiosity.

Same-model re-runs sitting at exactly 1.0000 on both numbers is what sets the
0.99 floor: on this provider there is no float drift to leave room for, so
anything below 0.99 is a real change rather than noise.

## The probes

Twelve fixed texts, frozen in [`probes.py`](embed_drift/probes.py) and hashed
into every fingerprint. A check against a fingerprint built from a different set
**refuses to run** rather than reporting a number that means nothing.

They are spread across what a revision tends to break unevenly: two
near-paraphrases that must stay close, two unrelated sentences that must stay
apart, code, SQL, German, Japanese, a long paragraph near the truncation limit,
punctuation-only text, a bare number, and a single space. Twelve gives 66
pairwise distances — a stable enough shape to correlate, and cheap enough that a
check costs twelve requests.

## Honest limits

- **A fingerprint is not a guarantee, it is a tripwire.** It tells you the space
  moved. It cannot tell you whether the new model is better or worse for your
  corpus — only a retrieval evaluation on your own data answers that.
- **The probes are generic on purpose.** A model revision that degrades only on
  your domain's vocabulary can pass this and still hurt you. Twelve probes of
  your own content alongside these would catch more; the probe set is one file.
- **Thresholds are defaults, not truths.** `0.99` on both numbers is tuned to a
  same-model re-run on Ollama, which sits at 1.0. A provider that rounds its
  output more coarsely will need a looser floor, measured rather than guessed.
- **One request per probe, no batching.** A batch endpoint that silently
  truncates or reorders would corrupt a fingerprint in a way that looks exactly
  like drift, and not crying wolf matters more here than twelve saved requests.
- **A drifted verdict does not say who changed it.** A provider revision, a
  different quantisation of the same local model, or someone editing the model
  name in a config all look the same from outside.

## Reproduce it

```bash
ollama pull all-minilm
ollama pull nomic-embed-text

py -m embed_drift.cli baseline --model all-minilm --out fp.json
py -m embed_drift.cli check fp.json                            # same model
py -m embed_drift.cli check fp.json --model nomic-embed-text   # a different one
py -m unittest discover -s tests
```

## Licence

MIT, © Dmytro Galko. See [LICENSE](LICENSE).
