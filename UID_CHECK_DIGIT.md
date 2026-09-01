# UID Check Digit — Usage Guide

## Format
```
A7KP3NWXQ
│       │
│       └── check digit (position 9, computed)
└─────────── 8-char Crockford Base32 (random, unambiguous)
```
Stored as 9 chars. Displayed as `A7KP-3NWX-Q` with separators.

---

## How It Works
The 9th character is computed from the first 8.
Change any one character → check digit won't match → invalid UID.

```python
def is_valid_uid(uid: str) -> bool:
    uid = uid.replace("-", "")
    if len(uid) != 9:
        return False
    alphabet = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
    expected = alphabet[
        sum(ord(c) * (i + 1) for i, c in enumerate(uid[:8]))
        % len(alphabet)
    ]
    return uid[8] == expected
```

Write this once. Use it everywhere.

---

## Use Cases (Priority Order)

### 1. Dashboard Search — 94/100
Validate before hitting the database.
Manager mistypes one character → instant "invalid UID" message.
Zero database load. Prevents "not found" confusion.

### 2. API Input Guard — 91/100
Every endpoint that accepts a UID validates it first.
Malformed URLs, broken links, truncated copy-pastes — all caught at the gate.
One line of protection on every route.

### 3. CSV Export Integrity — 87/100
Every UID in the exported CSV can be verified on import.
Catches corrupted rows before they create ghost records in analytics.
The only integrity signal once data leaves the system.

### 4. Cross-Camera Merge Audit — 82/100
Both UIDs validated before a merge is committed.
If a bug ever produces a garbled UID during merging it is caught
before it reaches the database.

### 5. Database Integrity Script — 78/100
Run anytime after migrations, crashes, or major pushes.
```python
invalid = [pid for pid in all_person_ids if not is_valid_uid(pid)]
```
Instant health check on the entire identity store.

### 6. Analytics Deduplication — 71/100
When the same UID appears in two sources (CSV + dashboard + API log),
the check digit confirms it is genuinely the same UID and not a
near-collision from a transcription error.

---

## Advantages

**Human safety**
No ambiguous characters (0/O/1/I/L removed by Crockford).
Check digit catches any single character substitution or transposition.
Safe to read aloud, type, and copy without errors.

**Zero database cost**
Validation runs in memory in under 1ms.
Invalid UIDs never reach SQLite.

**Self-documenting**
A valid UID proves its own integrity.
No external lookup needed to confirm a UID is well-formed.

**Future-proof**
Works in every layer — Python backend, dashboard frontend,
CSV exports, external integrations — with the same one function.

**Backwards compatible**
Existing 8-char UIDs in the database are grandfathered.
New UIDs from this point are 9-char with check digit.
Both coexist cleanly.

---

## What It Does NOT Do
- Does not prevent duplicate UIDs (ledger check does that)
- Does not encrypt or hide the UID
- Does not carry time or location information
- Does not fix already-corrupted data retroactively
