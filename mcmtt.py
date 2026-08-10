"""
mcmtt.py — Multi-Camera Multi-Target Tracking core.

Design
------
* GlobalGallery   : centralised store of global UIDs -> multi-view appearance
                    "prototypes" (clustered embeddings), plus occupancy state
                    (which camera each UID was last seen on, and when).
* Spatiotemporal gate: a UID that is ACTIVE on another camera within
                    CO_OCCURRENCE_WINDOW seconds is INELIGIBLE for a detection
                    on this camera, unless enough time has passed for a real
                    person to physically walk between the two views
                    (MIN_TRANSIT_TIME).
* Hungarian assignment: all detections in a frame are matched against all
                    eligible gallery identities in ONE global optimal
                    assignment, so no identity is claimed twice.
* Multi-view prototypes: each identity keeps up to N_PROTOTYPES cluster
                    centroids rather than one mean, so a person seen sitting,
                    standing, front-on and from behind still matches. This is
                    what gives robustness to viewpoint / illumination change.

Pure NumPy; scipy is used for Hungarian if present, else an internal
implementation is used.
"""
from __future__ import annotations

import time
import numpy as np

try:  # scipy is optional
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False


# ---------------------------------------------------------------- utilities
def l2n(v):
    """L2-normalise a vector (or rows of a matrix)."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    return v / n


def _hungarian_numpy(cost):
    """Minimal O(n^3) Hungarian (Jonker-Volgenant style augmenting path).
    Used only when scipy is unavailable. Returns (rows, cols)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = m, n
        transposed = True
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def hungarian(cost):
    """Optimal assignment minimising total cost. Returns (row_idx, col_idx)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if _HAVE_SCIPY:
        return _scipy_lsa(cost)
    return _hungarian_numpy(cost)


# ------------------------------------------------------------------ identity
class Identity:
    """A global identity: multi-view appearance prototypes + occupancy state."""

    __slots__ = ("uid", "protos", "counts", "first_seen", "last_seen",
                 "last_cam", "cam_last_seen", "n_updates")

    def __init__(self, uid, vec, cam, now=None):
        now = now if now is not None else time.time()
        self.uid = uid
        self.protos = [l2n(vec)]      # list of prototype vectors (views)
        self.counts = [1]             # how many samples formed each prototype
        self.first_seen = now
        self.last_seen = now
        self.last_cam = cam
        self.cam_last_seen = {cam: now} if cam else {}
        self.n_updates = 1

    # -- appearance -------------------------------------------------------
    def similarity(self, vec):
        """Max cosine similarity against any prototype (multi-view match)."""
        if not self.protos:
            return 0.0
        P = np.asarray(self.protos, dtype=np.float32)
        return float(np.max(P @ l2n(vec)))

    def update(self, vec, cam, now=None, merge_thr=0.75, max_protos=6):
        """Fold a new observation in. If it is close to an existing prototype,
        update that prototype (running mean); otherwise add a NEW prototype so a
        different viewpoint/illumination is represented rather than averaged
        away. This is what keeps a person matchable after they stand up/turn."""
        now = now if now is not None else time.time()
        v = l2n(vec)
        P = np.asarray(self.protos, dtype=np.float32)
        sims = P @ v
        j = int(np.argmax(sims))
        if float(sims[j]) >= merge_thr:
            c = self.counts[j]
            self.protos[j] = l2n((self.protos[j] * c + v) / (c + 1))
            self.counts[j] = c + 1
        elif len(self.protos) < max_protos:
            self.protos.append(v)
            self.counts.append(1)
        else:
            # replace the least-supported prototype
            k = int(np.argmin(self.counts))
            self.protos[k] = v
            self.counts[k] = 1
        self.last_seen = now
        self.n_updates += 1
        if cam:
            self.last_cam = cam
            self.cam_last_seen[cam] = now

    # -- occupancy --------------------------------------------------------
    def active_on_other_camera(self, cam, now, window):
        """True if this identity was seen on a DIFFERENT camera within `window`
        seconds — i.e. it is currently occupied elsewhere."""
        for c, ts in self.cam_last_seen.items():
            if c != cam and (now - ts) <= window:
                return True, c, now - ts
        return False, None, None

    def cameras(self):
        return set(self.cam_last_seen.keys())


# ------------------------------------------------------------------ gallery
class GlobalGallery:
    """Centralised gallery of global UIDs with spatiotemporal-aware matching."""

    def __init__(self,
                 same_cam_threshold=0.68,
                 cross_cam_threshold=0.80,
                 co_occurrence_window=2.0,
                 min_transit_time=3.0,
                 max_protos=6,
                 proto_merge_thr=0.75,
                 ambiguity_margin=0.05,
                 max_identities=5000,
                 overlap_groups=None):
        self.same_cam_threshold = float(same_cam_threshold)
        self.cross_cam_threshold = float(cross_cam_threshold)

        self.co_occurrence_window = float(co_occurrence_window)
        # Even after the co-occurrence window, a person needs physical time to
        # walk between views. Matches faster than this are rejected.
        self.min_transit_time = float(min_transit_time)
        self.max_protos = int(max_protos)
        self.proto_merge_thr = float(proto_merge_thr)
        self.ambiguity_margin = float(ambiguity_margin)
        self.max_identities = int(max_identities)

        self.overlap_groups = [set(g) for g in (overlap_groups or [])]
        self._ids = {}     # uid -> Identity
        self.stats = {"assigned": 0, "created": 0, "blocked_cooccurrence": 0,
                      "blocked_transit": 0, "ambiguous": 0}

    def _cameras_overlap(self, cam_a, cam_b):
        """True if two cameras share a physical space (same overlap group), so
        the same person may legitimately appear on both at once."""
        if cam_a == cam_b:
            return True
        for g in self.overlap_groups:
            if cam_a in g and cam_b in g:
                return True
        return False

    # -- basic accessors --------------------------------------------------
    def __len__(self):
        return len(self._ids)

    def get(self, uid):
        return self._ids.get(uid)

    def cameras_of(self, uid):
        idn = self._ids.get(uid)
        return idn.cameras() if idn else set()

    def add(self, uid, vec, cam, now=None):
        self._ids[uid] = Identity(uid, vec, cam, now)
        self.stats["created"] += 1
        return self._ids[uid]

    def update(self, uid, vec, cam, now=None):
        idn = self._ids.get(uid)
        if idn is None:
            return self.add(uid, vec, cam, now)
        idn.update(vec, cam, now, self.proto_merge_thr, self.max_protos)
        return idn

    def drop(self, uid):
        """Forget a UID entirely (it was merged away in the DB)."""
        return self._ids.pop(uid, None) is not None

    def merge(self, keep_uid, drop_uid):
        """Fold drop_uid's prototypes and occupancy into keep_uid, then forget
        drop_uid. Must be called whenever PersonDB.merge_persons() succeeds,
        otherwise the dead UID stays in the gallery: it keeps matching incoming
        vectors, and — worse — its stale cam_last_seen entries make the SURVIVOR
        look like it is co-occurring with itself, so _eligible() blocks the
        legitimate cross-camera hand-off and the person gets a second UID."""
        if keep_uid == drop_uid:
            return False
        d = self._ids.pop(drop_uid, None)
        if d is None:
            return False
        k = self._ids.get(keep_uid)
        if k is None:
            d.uid = keep_uid
            self._ids[keep_uid] = d
            return True
        # fold appearance prototypes (respects max_protos / merge threshold)
        for p, c in zip(d.protos, d.counts):
            k.update(p, None, k.last_seen, self.proto_merge_thr, self.max_protos)
        # occupancy: keep the most recent sighting per camera
        for cam, ts in d.cam_last_seen.items():
            if ts > k.cam_last_seen.get(cam, 0.0):
                k.cam_last_seen[cam] = ts
        if d.last_seen > k.last_seen:
            k.last_seen = d.last_seen
            k.last_cam = d.last_cam
        k.first_seen = min(k.first_seen, d.first_seen)
        k.n_updates += d.n_updates
        return True

    # -- eligibility ------------------------------------------------------
    def _eligible(self, idn, cam, now):
        """Spatiotemporal gate. Returns (ok, reason).

        Overlapping-FOV cameras (same physical room) are EXEMPT from the
        co-occurrence and transit gates: the same person genuinely appears on
        both at once, so 'busy elsewhere' and 'too soon to have travelled' are
        not evidence of a different person there. Non-overlapping cameras keep
        both gates, so a fixture in another room still cannot steal a UID."""
        busy, other_cam, age = idn.active_on_other_camera(
            cam, now, self.co_occurrence_window)
        if busy and not self._cameras_overlap(cam, other_cam):
            return False, f"co-occurrence(on {other_cam} {age:.1f}s ago)"
        # physical transit feasibility (only for non-overlapping cameras)
        last_elsewhere = None
        for c, ts in idn.cam_last_seen.items():
            if c != cam and not self._cameras_overlap(cam, c):
                last_elsewhere = ts if last_elsewhere is None else max(last_elsewhere, ts)
        if last_elsewhere is not None:
            dt = now - last_elsewhere
            if dt < self.min_transit_time:
                return False, f"transit({dt:.1f}s < {self.min_transit_time}s)"
        return True, None

    # -- the main entry point --------------------------------------------
    def assign(self, detections, cam, now=None, reserved=None):
        """Globally assign a frame's detections to gallery identities.

        detections : list of dicts, each with at least {"vec": np.ndarray}.
                     Anything else (e.g. "tkey") is passed through untouched.
        cam        : camera name for this batch of detections.
        reserved   : set of UIDs already claimed by live tracks on this camera
                     (never reassign them to a different detection).

        Returns a list, parallel to `detections`, of dicts:
            {"uid": <uid or None>, "score": float, "reason": str}
        uid is None when the detection should become a NEW identity.
        """
        now = now if now is not None else time.time()
        reserved = set(reserved or ())
        out = [{"uid": None, "score": 0.0, "reason": "no-candidates"}
               for _ in detections]
        if not detections or not self._ids:
            return out

        # 1) Build the candidate set, applying the spatiotemporal gate ONCE.
        cand_uids, cand_ids, blocked = [], [], {}
        for uid, idn in self._ids.items():
            if uid in reserved:
                continue
            ok, reason = self._eligible(idn, cam, now)
            if not ok:
                blocked[uid] = reason
                if "co-occurrence" in reason:
                    self.stats["blocked_cooccurrence"] += 1
                else:
                    self.stats["blocked_transit"] += 1
                continue
            cand_uids.append(uid)
            cand_ids.append(idn)
        if not cand_uids:
            for r in out:
                r["reason"] = "all-candidates-blocked"
            return out

        # 2) Similarity matrix  [n_det x n_cand]  (max over prototypes).
        D = l2n(np.asarray([d["vec"] for d in detections], dtype=np.float32))
        S = np.zeros((len(detections), len(cand_uids)), dtype=np.float32)
        for j, idn in enumerate(cand_ids):
            P = np.asarray(idn.protos, dtype=np.float32)
            S[:, j] = np.max(D @ P.T, axis=1)

        # 3) Per-candidate threshold: same-camera vs cross-camera bar.
        bars = np.empty(len(cand_uids), dtype=np.float32)
        for j, idn in enumerate(cand_ids):
            seen_here = cam in idn.cam_last_seen
            bars[j] = (self.same_cam_threshold if seen_here
                       else self.cross_cam_threshold)

        # 4) Global optimal assignment (Hungarian) on cost = 1 - similarity.
        #    Pairs below their bar are made prohibitively expensive so the
        #    optimiser will not choose them.
        BIG = 1e6
        cost = 1.0 - S.astype(np.float64)
        cost[S < bars[None, :]] = BIG
        rows, cols = hungarian(cost)

        for r, c in zip(rows, cols):
            if cost[r, c] >= BIG:
                out[r]["reason"] = "below-threshold"
                out[r]["score"] = float(S[r].max()) if S.shape[1] else 0.0
                continue
            # ambiguity check: is the runner-up almost as good?
            row = S[r].copy()
            best = float(row[c])
            row[c] = -1.0
            second = float(row.max()) if row.size > 1 else 0.0
            if (best - second) < self.ambiguity_margin and second >= bars[int(np.argmax(row))]:
                self.stats["ambiguous"] += 1
                out[r] = {"uid": None, "score": best, "reason": "ambiguous"}
                continue
            out[r] = {"uid": cand_uids[c], "score": best, "reason": "matched"}
            self.stats["assigned"] += 1
        return out

    # -- housekeeping -----------------------------------------------------
    def prune(self, max_age=3600.0, now=None):
        """Drop identities unseen for `max_age` seconds; also cap total size."""
        now = now if now is not None else time.time()
        dead = [u for u, i in self._ids.items() if (now - i.last_seen) > max_age]
        for u in dead:
            self._ids.pop(u, None)
        if len(self._ids) > self.max_identities:
            order = sorted(self._ids.items(), key=lambda kv: kv[1].last_seen)
            for u, _ in order[:len(self._ids) - self.max_identities]:
                self._ids.pop(u, None)
        return len(dead)

    def health(self):
        """Separation diagnostics: within- vs cross-identity similarity."""
        uids = [u for u, i in self._ids.items() if len(i.protos) >= 2]
        within = []
        for u in uids:
            P = np.asarray(self._ids[u].protos, dtype=np.float32)
            G = P @ P.T
            iu = np.triu_indices(len(P), k=1)
            within.extend(G[iu].tolist())
        cross = []
        keys = list(self._ids.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                Pa = np.asarray(self._ids[keys[a]].protos, dtype=np.float32)
                Pb = np.asarray(self._ids[keys[b]].protos, dtype=np.float32)
                cross.append(float(np.max(Pa @ Pb.T)))
        return {
            "identities": len(self._ids),
            "within_mean": float(np.mean(within)) if within else None,
            "cross_mean": float(np.mean(cross)) if cross else None,
            "cross_p95": float(np.percentile(cross, 95)) if cross else None,
            **self.stats,
        }
