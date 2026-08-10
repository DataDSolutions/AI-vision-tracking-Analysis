import os

# ---------------------------------------------------------------------------
# Orin Nano VPI/PVA segfault mitigation. The run ends in a SIGSEGV inside the
# NvDCF visual feature extractor (GPUassert cuDCFv2/featureExtractor.cu) because
# the Orin Nano has NO PVA hardware ("Failed to open PVA device node" at startup)
# and any tracker op that reaches for a VPI/PVA backend crashes — more often when
# many objects are in view. The tracker config already keeps useVPICropScaler=0
# and visualTrackerType=1 (NvDCF, not NvDCF_VPI). This forces every remaining VPI
# op onto the CUDA backend (config mitigation #2), which must be set BEFORE the
# tracker's .so is dlopen'd — i.e. before any pyservicemaker / GStreamer import.
# Set VPI_DEFAULT_BACKEND=<something-else> in the env to override.
os .environ .setdefault ("VPI_DEFAULT_BACKEND","cuda")

import re
import csv
import time
import math
import ctypes
import threading
import numpy as np

from collections import defaultdict ,deque ,OrderedDict
from datetime import datetime

from pyservicemaker import Pipeline ,Probe ,BatchMetadataOperator
from pyservicemaker import osd as _osd

from person_db import PersonDatabase

CAMERA_STREAMS =[

{"name":"cam1-ch301","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/301"},
#{"name":"cam1-ch401","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/401"},
#{"name":"cam1-ch701","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/701"},
{"name":"cam2-ch801","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/801"},
{"name":"cam3-ch201","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/201"},
{"name":"cam4-ch501","uri":"rtsp://sanjay:prama8833@192.168.88.28:554/Streaming/channels/501"},
]

ONLY_CAMERA_INDEX =None
if ONLY_CAMERA_INDEX is not None :
    CAMERA_STREAMS =[CAMERA_STREAMS [ONLY_CAMERA_INDEX ]]

if os .environ .get ("USE_SUBSTREAM","0")not in ("0","false","False",""):
    _sub =[]
    for _c in CAMERA_STREAMS :
        _u =re .sub (r"(/channels/\d*)1(\b|/|$)",r"\g<1>2\g<2>",_c ["uri"])
        _sub .append ({"name":_c ["name"]+"-sub","uri":_u })
    if any (s ["uri"]!=c ["uri"]for s ,c in zip (_sub ,CAMERA_STREAMS )):
        print ("[SOURCE] USE_SUBSTREAM=1 -> using camera substreams:")
        for s in _sub :
            print (f"[SOURCE]   {s ['name']}: {s ['uri']}")
        CAMERA_STREAMS =_sub
    else :
        print ("[SOURCE] USE_SUBSTREAM set but no channel URL matched the "
        "main-stream pattern; leaving URIs unchanged. Edit CAMERA_STREAMS "
        "manually if your substream channels differ.")

_skip_env =os .environ .get ("SKIP_CAMERAS","").strip ()
if _skip_env and ONLY_CAMERA_INDEX is None :
    _skip =set ()
    for tok in _skip_env .replace (" ","").split (","):
        if tok .isdigit ():
            _skip .add (int (tok ))
    if _skip :
        kept =[c for i ,c in enumerate (CAMERA_STREAMS )if i not in _skip ]
        dropped =[CAMERA_STREAMS [i ]["name"]for i in sorted (_skip )
        if 0 <=i <len (CAMERA_STREAMS )]
        if dropped :
            print (f"[CAMERA] SKIP_CAMERAS={_skip_env } -> excluding: "
            f"{', '.join (dropped )}")
        if kept :
            CAMERA_STREAMS =kept
        else :
            print ("[CAMERA] SKIP_CAMERAS would drop ALL cameras — ignoring it.")

# labels.txt is the SINGLE SOURCE OF TRUTH for classes. The class list is NOT
# hardcoded anywhere in this file — change labels.txt (and num-detected-classes
# in config_infer.txt to match) and the app adapts automatically.
#
# Format: one class name per line, in class-id order (line 0 = class_id 0).
# Optional per-line control of person/non-person handling via a trailing tag:
#     Adult - Male sitting        # (default: auto-detected as person by keyword)
#     Dog                    !nonperson   # force: never tracked as a person
#     Robot inspector        !person      # force: tracked as a person
# Anything after '#' on a line is ignored as a comment.
LABELS_PATH =os .environ .get ("LABELS_PATH","labels.txt")

PERSON_KEYWORDS =("adult","kid","female","male","boy","girl",
"person","child","man","woman")
# Kept ONLY for the optional auto-tagging hint below; NOT used to reject a class.
NON_PERSON_KEYWORDS =("dog","cat","animal","vehicle","car","bag",
"object")

# Explicit overrides parsed from labels.txt (!person / !nonperson tags),
# keyed by class id. These are the ONLY way a labeled class is excluded.
_LABEL_PERSON_OVERRIDE ={}

# When True (default), EVERY class present in labels.txt is trackable unless it
# carries an explicit "!nonperson" tag. This makes labels.txt the single source
# of truth for what is tracked, and removes the old keyword heuristic that could
# silently discard a legitimately-labeled class the model was trained to detect.
# Set TRACK_ALL_LABELED=0 to fall back to keyword-based person matching.
TRACK_ALL_LABELED =os .environ .get ("TRACK_ALL_LABELED","1")not in ("0","false","False","")


def load_labels (path =None ):
    """Load class names from labels.txt. Fails loudly if the file is missing —
    it is the source of truth and must not be silently regenerated, or IDs
    would be assigned against the wrong class order."""
    path =path or LABELS_PATH
    if not os .path .isfile (path ):
        raise SystemExit (
        f"[LABELS] '{path }' not found. This file is the source of truth for "
        "class names and must exist. Create it with one class name per line, "
        "in class-id order (line 0 = class_id 0), matching your model and "
        "num-detected-classes in config_infer.txt. "
        "You can point elsewhere with the LABELS_PATH environment variable.")

    names =[]
    _LABEL_PERSON_OVERRIDE .clear ()
    with open (path )as f :
        for raw in f :
            line =raw .split ("#",1 )[0 ].rstrip ()
            if not line .strip ():
                continue
            override =None
            for tag in ("!nonperson","!non-person","!person"):
                if line .rstrip ().endswith (tag ):
                    override =(tag =="!person")
                    line =line [:line .rstrip ().rfind (tag )].rstrip ()
                    break
            name =line .strip ()
            if not name :
                continue
            if override is not None :
                _LABEL_PERSON_OVERRIDE [len (names )]=override
            names .append (name )
    if not names :
        raise SystemExit (f"[LABELS] '{path }' is empty — no class names found.")
    print (f"[LABELS] loaded {len (names )} classes from '{path }'"
    +(f" ({len (_LABEL_PERSON_OVERRIDE )} explicit person/non-person overrides)"
    if _LABEL_PERSON_OVERRIDE else ""))
    return names

LABELS =load_labels (LABELS_PATH )
NUM_CLASSES =len (LABELS )


def is_person_class (cid ):
    """Decide whether a class id is a trackable target.

    labels.txt is the SINGLE SOURCE OF TRUTH:
      1. An explicit !person / !nonperson tag in labels.txt always wins.
      2. Otherwise, in the default TRACK_ALL_LABELED mode, EVERY class that
         exists in labels.txt is trackable. This removes the old behaviour
         where a keyword like 'dog' or 'object' silently discarded a class the
         model was trained on — the user's requirement is that nothing labeled
         is falsely filtered.
      3. Only if TRACK_ALL_LABELED is disabled does it fall back to the legacy
         person-keyword heuristic.
    A class id outside labels.txt is never trackable (it is a config mismatch)."""
    if cid in _LABEL_PERSON_OVERRIDE :
        return _LABEL_PERSON_OVERRIDE [cid ]
    if not (0 <=cid <len (LABELS )):
        return False
    if TRACK_ALL_LABELED :
        return True
    low =LABELS [cid ].lower ()
    if any (k in low for k in NON_PERSON_KEYWORDS ):
        return False
    return any (k in low for k in PERSON_KEYWORDS )

POSTURE_WORDS =("sitting","standing")

def split_label (label ):

    low =label .lower ()
    for p in POSTURE_WORDS :
        if low .endswith (p ):
            return label [:len (label )-len (p )].rstrip ().rstrip ("-").rstrip (),p
    return label ,None

def compose_label (demo ,posture ):
    return f"{demo } {posture }"if posture else demo

# Census buckets used by the CSV summary columns. The 13 fine-grained labels
# (sitting/standing/etc.) collapse into four counted groups; Dog and anything
# unrecognised map to None and are NOT counted in the Total Unique columns.
DEMO_GROUPS =("Female","Male","Kid - Boy","Kid - Girl")
# Fraction of a UID's pooled demographic votes the leading group must hold before
# its label is shown. Below this the raw argmax is used. Protects against the
# detector's sporadic wrong-gender frames (male shown as Adult-Female) flipping a
# stable identity's displayed demographic. 0.60 = the leader needs a clear
# majority, not a mere plurality.
DEMO_MAJORITY =float (os .environ .get ("REID_DEMO_MAJORITY","0.60"))

def demo_group (demo ):
    """Map the demographic half of a label ('Adult - Female', 'Kid - Boy', ...)
    to one of DEMO_GROUPS, or None. Order matters: 'female' contains the
    substring 'male', so Female/Girl must be tested before Male/Boy."""
    if not demo :
        return None
    d =demo .lower ()
    is_kid =("kid"in d )or ("child"in d )
    if "girl"in d :
        return "Kid - Girl"
    if "boy"in d :
        return "Kid - Boy"
    if "female"in d :
        return "Kid - Girl"if is_kid else "Female"
    if "male"in d :
        return "Kid - Boy"if is_kid else "Male"
    return None

# ---- Demographic (age / gender) resolution ---------------------------------
# One identity = one demographic, pooled from the per-UID class votes.
#
# CORRECTNESS FIX (v26): the previous version summed gender across ages —
# "Female" = (Adult-Female votes + Kid-Girl votes), "Male" = (Adult-Male +
# Kid-Boy) — and took the larger bucket. On this detector, a genuinely
# Adult-Male person picks up scattered wrong frames of BOTH "Adult - Female" and
# "Kid - Girl"; those two female-ish buckets SUM and can beat the single, larger
# "Adult - Male" class, so the identity flipped to Female in the CSV. That is the
# observed "almost everything is female" bug. The fix is to take the demographic
# from the DOMINANT CONCRETE CLASS (what the detector actually emits most for
# this identity) rather than from cross-age gender sums.
#
# The ONE deliberate override kept from before is an age guard: the rarer,
# higher-cost "Kid" call is only shown when Kid votes hold a clear majority AND a
# minimum count, so an intermittent adult->kid mislabel (image: an adult drawn
# "Kid - Girl sitting") does not turn an adult into a child. Gender is then taken
# from the dominant class WITHIN the resolved age band, so it never mixes a
# Kid-Girl vote's gender onto an Adult-Male base.
#
# NOTE: this is a robustness layer over NOISY detector output — it cannot fix a
# model that emits the wrong class on (nearly) every frame for a person. If a
# person is STILL mislabelled after this, the fix is at the detector/threshold/
# training level; REID_KID_MIN_RATIO / REID_KID_MIN_VOTES tune the age guard.
KID_MIN_RATIO =float (os .environ .get ("REID_KID_MIN_RATIO","0.60"))
KID_MIN_VOTES =int (os .environ .get ("REID_KID_MIN_VOTES","3"))

def _demo_axes (label ):
    """Split a demographic label into (age, gender); each may be None.
    Order matters because 'female' contains 'male' and 'woman' contains 'man'."""
    d =(label or "").lower ()
    age =None
    if ("kid"in d )or ("child"in d ):
        age ="Kid"
    elif ("adult"in d )or ("woman"in d )or ("man"in d ):
        age ="Adult"
    gender =None
    if ("girl"in d )or ("female"in d )or ("woman"in d ):
        gender ="Female"
    elif ("boy"in d )or ("male"in d )or ("man"in d ):
        gender ="Male"
    return age ,gender

def _compose_demo (age ,gender ):
    """Rebuild a canonical demographic label from the two resolved axes."""
    if age =="Kid":
        if gender =="Female":
            return "Kid - Girl"
        if gender =="Male":
            return "Kid - Boy"
        return "Kid"
    # Adult or unknown age -> default to Adult when a gender is known.
    if gender =="Female":
        return "Adult - Female"
    if gender =="Male":
        return "Adult - Male"
    return "Person"

def resolve_uid_demo (pid ,fallback =""):
    """One identity = one demographic, from the DOMINANT concrete class (not from
    cross-age gender sums), with a conservative Kid age guard. Used for BOTH the
    on-screen label and the CSV flags so they always agree."""
    votes =uid_demo_votes .get (pid )
    if not votes :
        return fallback
    # Age guard: default Adult; declare Kid only on a clear, well-supported
    # majority of KID votes (resists the intermittent adult->kid mislabel).
    kid_n =adult_n =0
    for lbl ,n in votes .items ():
        a ,_g =_demo_axes (lbl )
        if a =="Kid":
            kid_n +=n
        elif a =="Adult":
            adult_n +=n
    agetot =kid_n +adult_n
    age ="Adult"
    if (agetot >0 and kid_n >=KID_MIN_VOTES
    and (kid_n /float (agetot ))>=KID_MIN_RATIO ):
        age ="Kid"
    # Gender from the dominant concrete class WITHIN the resolved age band, so a
    # larger single class wins and no cross-age gender summation can flip it.
    cand =[(lbl ,n )for lbl ,n in votes .items ()if _demo_axes (lbl )[0 ]==age ]
    if not cand :
        cand =list (votes .items ())
    base =max (cand ,key =lambda kv :kv [1 ])[0 ]
    _a ,gender =_demo_axes (base )
    demo =_compose_demo (age ,gender )
    return demo or fallback

FRAME_W =1280
FRAME_H =720

# ReID cosine-similarity gates (env-tunable for on-device tuning without edits).
# Cross-camera similarity is typically LOWER than same-camera (lighting/pose
# differ between cam7 outdoor and cam8 indoor), so the cross-cam bar is looser.
# If you see wrong merges, raise CROSS_CAMERA_THRESHOLD; if hand-offs are missed,
# lower it. Watch the [REID-HEALTH] "stranger floor" vs "same person" margin.
SIMILARITY_THRESHOLD =float (os .environ .get ("REID_SAME_CAM_THR","0.68"))
SOFT_THRESHOLD =float (os .environ .get ("REID_SOFT_THR","0.62"))
# Measured on-device: stranger (cross-identity) p95 ~0.64-0.78, same-person mean
# ~0.97. A 0.55 bar sat BELOW the stranger p95, so different people were being
# merged (observed [X-CAM] merges at 0.619 / 0.643). 0.80 sits above the stranger
# p95 and well below the same-person mean, giving a safe separation.
# Watch the [REID-HEALTH] "stranger floor" line and keep this above its p95.
# 0.84. The previous 0.88 blocked EVERY cross-camera hand-off for a whole run
# ("hand-offs succeeded: 0", "NO UID has been seen on more than one camera")
# while the reported stranger p95 still climbed to 0.99 — proof that the bar
# was not the thing separating identities. That inflated p95 was an ARTIFACT:
# commits were failing, so one real person existed as several UIDs and got
# counted as "different people" scoring ~1.0 against each other. With the
# commit path fixed that pollution goes away. 0.84 sits below every observed
# within-identity mean (0.841-0.974) and far above the stranger MEAN
# (0.52-0.64). Re-check the [REID-HEALTH] stranger floor after this run and
# tune from the real numbers.
# 0.84 strict bar (reverted from the 0.86 over-correction, which rejected the
# orange man's real 0.846 cross-camera match and minted a new UID for him). The
# strict bar is now backed up by a MARGIN-GATED floor below, so pollution control
# comes from the contamination guard + margin logic, NOT from an ever-higher bar.
CROSS_CAMERA_THRESHOLD =float (os .environ .get ("REID_CROSS_CAM_THR","0.84"))
# Margin-gated cross-camera acceptance: a cross-camera candidate at/above this
# floor that also beats the runner-up by REID_CROSS_MARGIN is accepted even below
# the strict bar. This recovers the 0.846 hand-off (its runner-up was far lower)
# while still rejecting ambiguous stranger collisions (several close candidates).
CROSS_CAMERA_FLOOR =float (os .environ .get ("REID_CROSS_CAM_FLOOR","0.80"))
CROSS_MARGIN =float (os .environ .get ("REID_CROSS_MARGIN","0.10"))
# Relaxed SAME-CAMERA bar used only for a candidate seen on this camera within
# the last few seconds — the signature of a person changing posture (stand<->sit)
# whose appearance drops below the normal 0.68 bar (measured: the man who stood
# on cam4 then sat on cam3 scored 0.642 against his own standing gallery and was
# given a NEW UID). Recency + same-camera + clear top-margin make this safe; it
# never lowers the bar for strangers or for cross-camera matches. Keep it above
# the stranger p95 (~0.69-0.75 in the logs) so it cannot merge different people.
POSTURE_THRESHOLD =float (os .environ .get ("REID_POSTURE_THR","0.60"))
POSTURE_WINDOW_SEC =float (os .environ .get ("REID_POSTURE_WINDOW","12.0"))
# Minimum centroid similarity for the same-camera duplicate reaper to collapse
# two live UIDs into one. Must be at/above the same-cam bar AND above the
# (possibly polluted) stranger p95, so it only ever fuses a genuine split of one
# person, never two different people. 0.72 sits above the observed 0.68 same-cam
# bar and the ~0.70 clean stranger p95.
DUP_REAP_MIN_SIM =float (os .environ .get ("REID_DUP_REAP_MIN_SIM","0.72"))
SOFT_WINDOW =1800.0
MAX_EMBEDDINGS =60

GALLERY_SEED_TARGET =25
GALLERY_SEED_INTERVAL =0.08
GALLERY_UPDATE_INTERVAL =1.0

REID_FEATURE_SIZE =256
TRACKER_CONFIG ="config_tracker_reid.yml"

# --- ReID metadata type ids (NvDsMetaType, nvdsmeta.h) ------------------------
# In DS 7.0+/9.1 the tracker attaches ReID as USER metadata, not via any
# "obj_reid_items" accessor. Each detected object carries an
# NVDS_TRACKER_OBJ_REID_META item (an NvDsObjReid struct: featureSize + vector,
# or on some builds an int index into the batch tensor). The whole-batch tensor
# is attached to the batch as NVDS_TRACKER_BATCH_REID_META.
#
# These integer ids are stable in nvdsmeta.h. They can be overridden from the
# environment if a future build renumbers them, without touching code.
NVDS_TRACKER_PAST_FRAME_META =int (os .environ .get ("NVDS_TRACKER_PAST_FRAME_META","5"))
NVDS_TRACKER_BATCH_REID_META =int (os .environ .get ("NVDS_TRACKER_BATCH_REID_META","6"))
NVDS_TRACKER_OBJ_REID_META =int (os .environ .get ("NVDS_TRACKER_OBJ_REID_META","7"))

EMB_DECIDE_COUNT =3
EMB_DECIDE_TIMEOUT =2.0

# --- Two-tier identity commitment -------------------------------------------
# Tier 1 (local): a confirmed per-camera track gets a GLOBAL UID as soon as it
# survives COMMIT_AFTER_HITS frames, WHETHER OR NOT a ReID vector exists yet.
# This guarantees within-camera identity persistence even when the ReID engine
# is sparse or silent. If a ReID vector is available at commit time we try to
# re-identify against the gallery first; otherwise we mint a fresh UID and
# enrich it with ReID vectors later as they arrive.
# Tier 2 (cross-camera): whenever a committed track later gets ReID vectors, it
# is matched against OTHER cameras' recent identities for hand-off/merge.
COMMIT_AFTER_HITS =6          # frames a track must survive before it gets a UID
REID_MERGE_MIN_VECS =2        # vectors needed before attempting a cross-cam merge
# Seconds between cross-camera merge attempts PER TRACK. Each attempt runs a
# full gallery match, so with many live tracks a low value dominates CPU and
# collapsed the pipeline to ~1 fps in testing. 5s is ample for hand-off.
REID_MERGE_INTERVAL =float (os .environ .get ("REID_MERGE_INTERVAL","5.0"))
MERGE_RETRY_BACKOFF =60.0     # seconds before retrying a merge pair that failed
_merge_failures ={}           # (keep,drop) -> last failure timestamp

# --- MCMTT: centralised multi-camera gallery ---------------------------------
# Enforces the physical constraint that ONE identity cannot be on TWO cameras at
# the same moment (this is what let an empty chair on cam-05 steal the UID of a
# person sitting on cam-04), and performs GLOBAL (Hungarian) assignment so two
# detections can never claim the same identity.
from mcmtt import GlobalGallery ,l2n as _mc_l2n

CO_OCCURRENCE_WINDOW =float (os .environ .get ("CO_OCCURRENCE_WINDOW","2.0"))
# was 3.0. The MCMTT transit gate rejects a cross-camera match that happens
# sooner than this after the identity was last seen (for real) on another
# camera. cam-07 and cam-08 face OPPOSITE directions along the same narrow
# corridor, so a person exits one view and enters the other in ~1-2s; a 3s gate
# rejected that legitimate hand-off and minted a new UID. 1s still blocks a
# physically-impossible instant teleport between truly distant cameras while
# allowing a real corridor crossing. Genuine simultaneous presence is separately
# blocked by the presence-gated live-track occupancy guard.
MIN_TRANSIT_TIME =float (os .environ .get ("MIN_TRANSIT_TIME","1.0"))
MAX_VIEW_PROTOS =int (os .environ .get ("MAX_VIEW_PROTOS","6"))
# OVERLAPPING-FOV CAMERA GROUPS. Cameras listed together here view the SAME
# physical space and legitimately see the same person at the same instant, so
# the MCMTT co-occurrence / min-transit gates are SKIPPED between them (those
# gates assume one person can't be on two cameras at once, which is false for
# overlapping views and was blocking correct cross-camera merges — e.g. the
# standing man on cam4-ch501 who sits down and reappears on cam3-ch201 could
# not merge back). Cameras in DIFFERENT groups keep both gates, so a fixture in
# another room still cannot steal a UID. Format (env): groups separated by ';',
# camera names within a group separated by ','. Example:
#   CAMERA_OVERLAP_GROUPS="cam3-ch201,cam4-ch501;cam1-ch301,cam2-ch801"
# Default below encodes the office/meeting-room pair seen in the sample images.
def _parse_overlap_groups (spec ):
    groups =[]
    for grp in (spec or "").split (";"):
        cams =[c .strip ()for c in grp .split (",")if c .strip ()]
        if len (cams )>=2 :
            groups .append (set (cams ))
    return groups
# DEFAULT: NO overlap group. "Overlapping FOV" means the SAME person is visible
# in two camera frames AT THE SAME INSTANT (the views physically share space). In
# this deployment cam1-ch401 / cam1-ch701 / cam3-ch201 / cam4-ch501 are DIFFERENT
# DESKS in the same office — each shows its OWN people, and no single person is in
# two views at once. Grouping them as overlapping (a bad v22 guess) DISABLED the
# co-occurrence exclusion between them, so one global UID (SFK6AT) landed on three
# DIFFERENT seated men simultaneously. With no group, the spatiotemporal gate is
# active on every camera pair: an identity still live on one camera cannot be
# claimed on another within CO_OCCURRENCE_WINDOW, which is exactly what stops a
# UID being painted on two different people at once.
#
# ONLY add cameras to a group if the SAME person is genuinely visible in both at
# the same moment (true FOV overlap). Format (env): groups separated by ';',
# cameras within a group by ','. Example for a real overlapping pair:
#   CAMERA_OVERLAP_GROUPS="cam3-ch201,cam4-ch501"
CAMERA_OVERLAP_GROUPS =_parse_overlap_groups (os .environ .get (
"CAMERA_OVERLAP_GROUPS",""))
GALLERY =None                 # created in main() once thresholds are known
ALLOW_REIDLESS_COMMIT =os .environ .get ("ALLOW_REIDLESS_COMMIT","1")not in ("0","false","False","")
# How long a confirmed track waits for a ReID vector before giving up and
# committing WITHOUT appearance. Committing immediately (the old behaviour, from
# when ReID was broken) minted a UID that was then merged away seconds later —
# visible as a person getting a UID, then none, then a different one. Now that
# ReID works, wait for an embedding so the FIRST UID is already the right one.
REIDLESS_COMMIT_DELAY =float (os .environ .get ("REIDLESS_COMMIT_DELAY","3.0"))

# --- Same-camera spatial ID recovery (anti-flicker) --------------------------
# The DeepStream tracker sometimes drops an object_id and re-mints a new one for
# the SAME physical person (posture change sit<->stand, brief occlusion, a
# detector gap). Without recovery that yields a brand-new UID = flicker. When a
# fresh tracker id appears, we first check whether a recently-lost UID existed
# at nearly the same place; if so the new track INHERITS that UID.
RECOVER_IOU =float (os .environ .get ("RECOVER_IOU","0.20"))
# ^ was 0.3. A seated person whose box shifts a little on re-detection (posture,
# lean, detector jitter) can drop below 0.30 IoU with its last box and then fail
# recovery -> new UID = flicker. 0.20 keeps the inheritance while still being
# well above coincidental overlap between two different people.
# --- Retention: how long an identity stays re-identifiable ------------------
# Requirement: a person's data is kept for 30 minutes so they can be recognised
# again on the SAME camera or a DIFFERENT one within that window.
# 48-hour rolling retention. Was 1800 (30 min), which meant anyone off-camera
# for half an hour came back as a BRAND NEW UID — the log showed "pruned 28
# identities unseen for >30 min" repeatedly. 48h keeps embeddings available for
# the full requirement window so a person who leaves and returns the next day
# still re-identifies to the same global UID.
RETENTION_SECONDS =float (os .environ .get ("RETENTION_SECONDS","172800"))  # 48 h
# Same-camera spatial recovery: a track that vanishes and reappears at nearly
# the same place inherits its UID without needing appearance. 6s was too short
# for someone who steps out of frame briefly; 45s covers normal comings/goings
# while still being far below the appearance-based retention window.
RECOVER_MAX_GAP =float (os .environ .get ("RECOVER_MAX_GAP","45.0"))
RECOVER_MAX_CENTER_FRAC =float (os .environ .get ("RECOVER_MAX_CENTER_FRAC","0.18"))
# ^ was 0.12. OR-path to IoU: inherit the lost UID if the new box's centre is
# within this fraction of the frame diagonal of the lost track's centre. Raised
# so a person who shifts in a chair (small centre move) still re-inherits rather
# than minting a new UID. Two distinct people rarely occupy the same centre.
# recent_local[(src_idx)] -> list of (pid, last_box, last_center, last_seen)
recent_local_tracks =defaultdict (list )

MIN_DET_CONFIDENCE =float (os .environ .get ("MIN_DET_CONFIDENCE","0.30"))
MIN_DET_CONFIDENCE_KEEP =float (os .environ .get ("MIN_DET_CONFIDENCE_KEEP","0.20"))
MIN_TRACK_HITS =5

STATIC_FLAG_ENABLE =True
STATIC_MIN_SECONDS =4.0
STATIC_MAX_DRIFT_PX =8.0
STATIC_MIN_SAMPLES =20

# Furniture / phantom suppression: a detection that is static from its very first
# frame and NEVER moves is almost certainly furniture (empty chair) mis-detected
# as a person, not a real person who walked in and sat down. We refuse to commit
# a NEW UID for such a track. A real person is normally seen moving in first, so
# they will have shown motion before this timeout elapses. Set
# SUPPRESS_STATIC_NEW=0 to disable if it ever hides a real motionless person.
SUPPRESS_STATIC_NEW =os .environ .get ("SUPPRESS_STATIC_NEW","1")not in ("0","false","False","")
STATIC_NEW_REJECT_SECONDS =8.0   # if static & motionless this long before commit -> furniture
# Minimum cumulative center displacement (px) since a track first appeared for it
# to count as "a person who moved" (vs furniture). Works from frame 1, so it
# catches empty chairs before they can commit. TRADE-OFF: a real person who is
# ALREADY seated when the stream starts and never moves is indistinguishable from
# furniture without ReID, so they will not get a UID until they move. Lower this
# if genuinely-still people are being missed; raise it if chairs slip through.
STATIC_MOVE_MIN_PX =float (os .environ .get ("STATIC_MOVE_MIN_PX","28"))
# Peak-displacement floor for the APPEARANCE-ONLY still-person commit escape
# (the branch that lets a genuinely-still seated person commit on a clean, non-
# fixture embedding without full motion confirmation). Set BELOW STATIC_MOVE_MIN_PX
# (the "clearly walked" bar): a seated person's shifting/leaning moves the box
# centre more than this over FIXTURE_LEARN_SECONDS, while an empty chair does not.
# This is the peak-travel alternative to the micro-motion-frame count; either one
# passing is enough evidence of a living occupant. Raise if chairs still slip
# through this escape; lower if a very still person is missed.
STATIC_STILL_PERSON_MIN_PX =float (os .environ .get ("STATIC_STILL_PERSON_MIN_PX","12"))
_track_first_center ={}
tracker_ever_moved =defaultdict (bool )
tracker_first_seen_ts ={}

# ---- MOTION-HISTORY personhood (v21) ---------------------------------------
# The decisive signal separating a PERSON from a FIXTURE (parked bikes, empty
# chair) is MOTION over the track's life — NOT detector confidence (a well-lit
# bike row detects with higher, steadier confidence than a dim seated person,
# so the v19/v20 confidence gate committed bikes and starved people) and NOT
# box deformation (a rocking chair deforms more than a still person).
#
# We accumulate, per track:
#   * peak displacement from birth position (_track_peak_disp)
#   * a count of frames showing real frame-to-frame motion (tracker_motion_frames)
# A person racks these up (arriving, leaning, reaching, shifting, standing); a
# rigid fixture stays pinned and accumulates ~none.
_track_last_center ={}
_track_peak_disp =defaultdict (float )
tracker_motion_frames =defaultdict (int )
_track_motion_run =defaultdict (int )
_track_center_win =defaultdict (lambda :deque (maxlen =16 ))
# Raised 9 -> 14. A net half-window shift below ~14px on a ~26fps stream is
# within the range an unstable detector box on a rigid object (empty chair,
# bike) produces, so at 9px chair-wobble was being counted as person motion and
# an office chair could accumulate MOTION_FRAMES_GRACE crossings and commit. 14
# requires a real translation of the box centre, which furniture does not do.
MOTION_STEP_MIN_PX =float (os .environ .get ("MOTION_STEP_MIN_PX","14"))
MOTION_FRAMES_FOR_PERSON =int (os .environ .get ("MOTION_FRAMES_FOR_PERSON","6"))
# The grace path (some-motion + past STATIC_GRACE_SECONDS -> commit) was the
# hole the empty chair fell through. Raised 4 -> 8 consecutive motion frames so
# only a track showing SUSTAINED translation (a person shifting/leaning) can use
# it; sporadic chair-jitter crossings never build a run this long.
MOTION_FRAMES_GRACE =int (os .environ .get ("MOTION_FRAMES_GRACE","8"))
STATIC_GRACE_SECONDS =float (os .environ .get ("STATIC_GRACE_SECONDS","4.0"))

# ---- AUTO-LEARNED FIXTURE ZONES (v21) --------------------------------------
# No bounding-box metric (motion, deformation, confidence, spread) can cleanly
# separate a nearly-still person from a false-positive on furniture/bikes whose
# detector box jitters — a jittery FP box mimics small human motion. The
# separating fact is PERSISTENCE-IN-PLACE: parked bikes and empty chairs occupy
# the SAME image region continuously, for the whole run. A person confirmed by
# MOTION (walking in, shifting, standing) is immune; a detection that is pinned
# to one spot for a long time REGISTERS that spot as a fixture zone, and later
# pinned detections there are suppressed. A person walking through moves out of
# the zone, so they are never suppressed by it.
#   key: (src_idx, gx, gy) grid cell; value: last time a pinned track was there
_fixture_zones ={}                      # {(src_idx, gx, gy): last_seen_ts}
FIXTURE_GRID_PX =float (os .environ .get ("FIXTURE_GRID_PX","64"))
# A non-moving track older than this REGISTERS its cell as a fixture zone.
# Lowered 10 -> 6. A pinned empty chair must REGISTER its zone and learn its
# appearance BEFORE the still-person commit branch (age >= this) can fire, or a
# brand-new chair could commit in the gap. 6s is still far longer than any
# detector flicker, so it never mislabels a walking-through person (who has left
# the cell — and latched ever_moved — long before 6s).
FIXTURE_LEARN_SECONDS =float (os .environ .get ("FIXTURE_LEARN_SECONDS","6.0"))
# A fixture zone is forgotten if no pinned track refreshes it for this long
# (so a chair that gets permanently occupied by a person eventually frees up).
FIXTURE_ZONE_TTL =float (os .environ .get ("FIXTURE_ZONE_TTL","120.0"))
# Fraction of a box's centre-cell neighbourhood that must be a fixture zone to
# suppress it. Kept simple: exact cell + immediate check.
FIXTURE_CONFIRM_SECONDS =float (os .environ .get ("FIXTURE_CONFIRM_SECONDS","30.0"))
FIXTURE_MAX_MOTION_FRAMES =int (os .environ .get ("FIXTURE_MAX_MOTION_FRAMES","3"))
# Box-deformation history (diagnostic only now; no longer commits a UID).
tracker_size_hist =defaultdict (lambda :deque (maxlen =90 ))
tracker_deform ={}
tracker_conf_hist =defaultdict (lambda :deque (maxlen =120 ))
DEFORM_MIN_SAMPLES =int (os .environ .get ("DEFORM_MIN_SAMPLES","30"))
DEFORM_PERSON_MIN =float (os .environ .get ("DEFORM_PERSON_MIN","0.002"))
DEFORM_MIN_SMOOTHNESS =float (os .environ .get ("DEFORM_MIN_SMOOTHNESS","0.30"))
DEFORM_CONFIRM_FRAMES =int (os .environ .get ("DEFORM_CONFIRM_FRAMES","12"))
# Legacy confidence-gate constants kept for the diagnostic report only.
PERSON_CONF_MEAN =float (os .environ .get ("PERSON_CONF_MEAN","0.60"))
PERSON_MIN_HITS =int (os .environ .get ("PERSON_MIN_HITS","20"))
PERSON_HIT_HZ =float (os .environ .get ("PERSON_HIT_HZ","8.0"))
PERSON_CONFIRM_STREAK =int (os .environ .get ("PERSON_CONFIRM_STREAK","8"))
# How long a committed UID stays reserved to its track after the track was
# last seen. Must be >= the tracker's maxShadowTrackingAge so a briefly-occluded
# or detector-missed person never has their identity handed to someone else.
# Raised 6s -> 18s to match maxShadowTrackingAge=450 (~17s) in the new tracker
# config: NvDCF now holds a seated/occluded person in shadow for ~17s, so their
# UID must stay reserved for at least that long or the UID could be reclaimed
# while the very same track is still alive in shadow.
UID_HOLD_SECONDS =float (os .environ .get ("UID_HOLD_SECONDS","18.0"))
# Cross-camera occupancy window. A committed UID counts as "still occupying"
# another camera (and is therefore ineligible for a hand-off here) ONLY if that
# camera saw it with a REAL detection within this many seconds. This is
# DELIBERATELY short and DECOUPLED from UID_HOLD_SECONDS (18s): UID_HOLD is a
# SAME-camera reservation sized to the tracker's ~17s shadow window, but reusing
# it for cross-camera exclusion meant a person who walked from cam-07 to cam-08
# was blocked from re-identifying to their own UID for ~18s (their cam-07 track
# was still coasting in shadow), so cam-08 minted a brand-new UID. 2s only blocks
# GENUINE simultaneous presence (both cameras seeing confident tracks at once,
# which is what the SFK6AT "one UID on three men" guard needs); a person who has
# actually left cam-07 is eligible on cam-08 within 2s. Raise only if real
# same-instant double-assignment reappears.
CROSS_CAM_BUSY_WINDOW =float (os .environ .get ("CROSS_CAM_BUSY_WINDOW","2.0"))
# A track whose NvDCF tracker_confidence is at/above this is treated as a REAL,
# present detection; below it (typical while coasting in shadow after the person
# left frame) it does not refresh the cross-camera occupancy timestamp. Builds
# that don't expose a real per-object confidence report ~1.0 always, in which
# case the short CROSS_CAM_BUSY_WINDOW alone still fixes the hand-off.
PRESENCE_CONF =float (os .environ .get ("PRESENCE_CONF","0.5"))
_deform_streak =defaultdict (int )


def _fixture_cell (src_idx ,cx ,cy ):
    return (src_idx ,int (cx //FIXTURE_GRID_PX ),int (cy //FIXTURE_GRID_PX ))


def _register_fixture (src_idx ,cx ,cy ,now ):
    """Mark this location as a persistent non-person fixture (bike row / empty
    chair). Called when a track has stayed pinned here long enough to be sure."""
    _fixture_zones [_fixture_cell (src_idx ,cx ,cy )]=now


def _in_fixture_zone (src_idx ,cx ,cy ,now ):
    """True if this location is a known fixture zone that is still fresh. Expired
    zones are pruned lazily so a spot that stops producing pinned detections
    (e.g. a chair a person now permanently occupies) is released."""
    cell =_fixture_cell (src_idx ,cx ,cy )
    ts =_fixture_zones .get (cell )
    if ts is None :
        return False
    if now -ts >FIXTURE_ZONE_TTL :
        _fixture_zones .pop (cell ,None )
        return False
    return True


def _deformation_score (tkey ,w ,h ):
    """Decide whether a track's bounding box behaves like a PERSON (non-rigid,
    smoothly changing) or an OBJECT (rigid). Returns (score, enough_samples).

    Two independent signals must agree, because either alone is foolable:

      1. MAGNITUDE - median absolute deviation of w/h relative to the median
         size. MAD rather than std-dev so a chair bumped ONCE, or a couple of
         detector spikes, cannot fake a person.
      2. SMOOTHNESS - lag-1 autocorrelation of the size signal. A person
         leaning or shifting traces a smooth trajectory (AC ~0.8-0.9); random
         detector jitter on a rigid object is white noise (AC ~0). This is
         what stops a noisy detector on an empty chair from reading as a
         person, which MAD alone could not.

    The returned score is the MAD magnitude gated by smoothness, so callers
    keep comparing against a single threshold (DEFORM_PERSON_MIN)."""
    hist =tracker_size_hist [tkey ]
    hist .append ((float (w ),float (h )))
    if len (hist )<DEFORM_MIN_SAMPLES :
        return 0.0 ,False
    wl =[a for a ,_ in hist ];hl =[b for _ ,b in hist ]
    n =len (wl )
    ws =sorted (wl );hs =sorted (hl )
    mw =ws [n //2 ];mh =hs [n //2 ]
    if mw <=1.0 or mh <=1.0 :
        return 0.0 ,True
    dw =sorted (abs (a -mw )for a in wl )
    dh =sorted (abs (b -mh )for b in hl )
    mag =((dw [n //2 ]/mw )+(dh [n //2 ]/mh ))/2.0

    def _ac1 (seq ):
        m =sum (seq )/len (seq )
        c =[x -m for x in seq ]
        den =sum (x *x for x in c )
        if den <=1e-9 :
            return 0.0
        return sum (c [i ]*c [i +1 ]for i in range (len (c )-1 ))/den

    smooth =max (_ac1 (wl ),_ac1 (hl ))
    if smooth <DEFORM_MIN_SMOOTHNESS :
        _deform_streak [tkey ]=0
        return 0.0 ,True          # jitter is white noise -> rigid object
    if mag <DEFORM_PERSON_MIN :
        _deform_streak [tkey ]=0
        return 0.0 ,True

    # PERSISTENCE. Lag-1 autocorrelation of white noise over N samples has
    # std ~1/sqrt(N) (~0.105 here), so a 0.30 bar is only ~2.9 sigma and is
    # crossed BY CHANCE roughly every 1500 frames — measured: every rigid
    # object committed at exactly 58.9s. A person satisfies both conditions
    # continuously, so requiring them to hold for consecutive frames costs a
    # real person a fraction of a second and makes a chance crossing
    # essentially impossible (p ~ 0.05^N).
    _deform_streak [tkey ]=_deform_streak .get (tkey ,0 )+1
    if _deform_streak [tkey ]<DEFORM_CONFIRM_FRAMES :
        return 0.0 ,True
    return mag ,True


_appearance_confirmed =set ()
_person_streak =defaultdict (int )


def _appearance_is_person (tkey ,conf ,age_seconds ,total_hits ):
    """Personhood from detector behaviour, independent of box motion.

    Returns (is_person, enough_evidence). A track qualifies as a person when the
    detector has fired on it enough times (PERSON_MIN_HITS) at a solid mean
    confidence (PERSON_CONF_MEAN), densely enough in time (PERSON_HIT_HZ hits per
    second, over the track's whole life). A real seated person clears this
    trivially — the detector locks onto them nearly every frame — while an
    empty-chair false positive does not: it flickers in and out at lower
    confidence.

    Density uses the MONOTONIC total_hits / age, NOT the length of the bounded
    confidence deque (which saturates at maxlen and would make a long-seated
    person's apparent hit-rate decay toward zero). Once confirmed, personhood is
    LATCHED: a person who later goes still or is briefly occluded stays a person.

    Independent of bounding-box shape, so it fixes BOTH observed failures: it
    rejects the rocking-but-empty chair that box-deformation accepted, and it
    accepts the perfectly-still seated people that box-deformation refused."""
    if tkey in _appearance_confirmed :
        return True ,True
    h =tracker_conf_hist [tkey ]
    if conf is not None :
        h .append (float (conf ))
    if total_hits <PERSON_MIN_HITS or len (h )==0 :
        return False ,False
    mean_conf =sum (h )/len (h )
    hit_hz =(total_hits /age_seconds )if age_seconds >0.5 else 0.0
    if mean_conf >=PERSON_CONF_MEAN and hit_hz >=PERSON_HIT_HZ :
        # Persistence: the mean-confidence bar must be cleared on
        # PERSON_CONFIRM_STREAK consecutive evaluations before we latch. A real
        # person sits well above the bar continuously; a borderline chair whose
        # windowed mean only momentarily drifts over 0.55 (measured ~3.5% of the
        # time at mean 0.50) does not sustain it. Same persistence trick that
        # removed the smoothness chance-crossing in the previous build.
        _person_streak [tkey ]=_person_streak .get (tkey ,0 )+1
        if _person_streak [tkey ]>=PERSON_CONFIRM_STREAK :
            _appearance_confirmed .add (tkey )
            return True ,True
        return False ,True
    _person_streak [tkey ]=0
    return False ,True

MIN_BOX_W =18
MIN_BOX_H =34
MIN_BOX_W_KEEP =12
MIN_BOX_H_KEEP =24

HEADLESS =os .environ .get ("HEADLESS","0")not in ("0","false","False","")
_NO_META_WRITE =os .environ .get ("NO_META_WRITE","0")not in ("0","false","False","")

# New-person aspect gate. Was 1.15 (box must be clearly taller than wide), which
# silently REJECTED crouching people and people whose lower body is occluded by
# motorbikes (cam-07/08) — their box is near-square. Lowered to 0.75 so a bent-
# over or partially-occluded person is accepted. Env-tunable per site.
MIN_PERSON_ASPECT_NEW =float (os .environ .get ("MIN_PERSON_ASPECT_NEW","0.75"))
MIN_PERSON_ASPECT_KEEP =float (os .environ .get ("MIN_PERSON_ASPECT_KEEP","0.65"))
ESTABLISH_AFTER_SEEN =3
MIN_PERSON_ASPECT =MIN_PERSON_ASPECT_NEW
MAX_PERSON_AREA_FRAC =0.35
PARTIAL_ASPECT_RATIO =1.6
POSTURE_WINDOW =30

TRACKER_TIMEOUT =300
CLEANUP_INTERVAL =60

DB_PATH ="reid_persons.db"

DB :PersonDatabase =None
tracker_to_person ={}
_track_emb_buffer ={}
tracker_last_seen ={}
# Last time each track was seen WITH A REAL, CONFIDENT detection (tracker
# confidence >= PRESENCE_CONF), as opposed to merely coasting in NvDCF shadow
# after the person has left the camera. The cross-camera occupancy guard reads
# THIS (not tracker_last_seen) so a person who walked out of cam-07 stops
# "occupying" cam-07 within CROSS_CAM_BUSY_WINDOW and can be handed off to cam-08.
tracker_present_ts ={}
camera_map ={}
tracker_class_votes =defaultdict (lambda :defaultdict (int ))
# Demographic (Adult-Male / Adult-Female / Kid-...) votes pooled PER COMMITTED
# UID rather than per tracker key. Posture legitimately differs between views
# (sitting on one camera, standing on another) but demographics must not: the
# same person was being drawn as "Adult - Male standing" on cam-02 and
# "Adult - Female standing" on cam-04 because each camera voted in isolation.
uid_demo_votes =defaultdict (lambda :defaultdict (int ))
tracker_posture_recent =defaultdict (lambda :deque (maxlen =POSTURE_WINDOW ))
tracker_hits =defaultdict (int )
tracker_seen =defaultdict (int )
tracker_positions =defaultdict (lambda :deque (maxlen =90 ))
tracker_is_static ={}
_last_gallery_update ={}

_cam_frames =defaultdict (int )
_cam_raw =defaultdict (int )
_cam_persons =defaultdict (int )
_cam_ids =defaultdict (set )
_cam_handoffs ={"ok":0 ,"missed":0 }
_src_idx_checked =False

_src_idx_seen =defaultdict (int )
_src_tag_seen =defaultdict (int )

_class_counts =defaultdict (int )
_class_rejected =defaultdict (int )
_unknown_cids =set ()

def class_health_report (final =False ):
    tag ="FINAL"if final else "REPORT"
    # Per-track appearance summary: how detector confidence and hit density
    # actually distribute for live tracks. This is what lets the PERSON_CONF_MEAN
    # / PERSON_HIT_HZ gates be tuned from real numbers instead of guessed. A
    # real person should show mean-conf well above the bar; an empty-chair false
    # positive should show a visibly lower mean-conf.
    _live =[k for k ,h in tracker_conf_hist .items ()if h ]
    if _live :
        confs =[sum (h )/len (h )for h in (tracker_conf_hist [k ]for k in _live )]
        confs .sort ()
        _committed =sum (1 for k in _live
        if not str (tracker_to_person .get (k ,"T")).startswith ("T"))
        _p =lambda q :confs [min (len (confs )-1 ,int (q *len (confs )))]
        print (f"[APPEARANCE {tag }] {len (_live )} live tracks "
        f"({_committed } committed, {len (_appearance_confirmed )} appearance-"
        f"confirmed) | mean-conf p10={_p (0.1 ):.2f} p50={_p (0.5 ):.2f} "
        f"p90={_p (0.9 ):.2f} | person gate: conf>={PERSON_CONF_MEAN } "
        f"hits>={PERSON_MIN_HITS } rate>={PERSON_HIT_HZ }/s")

    # Fixture-zone map: how many suppression zones have been auto-learned, and
    # how many live tracks are currently confirmed-person by motion. This is the
    # v21 furniture killer's observable state — a growing zone count over spots
    # like a bike row or an empty-chair cluster is the mechanism working.
    _moved =sum (1 for k in tracker_ever_moved if tracker_ever_moved [k ])
    print (f"[FIXTURES {tag }] {len (_fixture_zones )} auto-learned fixture zone(s) "
    f"| {_moved } track(s) motion-confirmed as person | learn "
    f"{FIXTURE_LEARN_SECONDS :.0f}s, grid {FIXTURE_GRID_PX :.0f}px, "
    f"TTL {FIXTURE_ZONE_TTL :.0f}s")

    if not _class_counts :
        return
    # BUGFIX: `total` was referenced (below) but never computed, so this
    # function raised 'name total is not defined' every report cycle. Because
    # class_health_report() runs inside the batch probe, that surfaced as the
    # recurring '[PROBE ERROR] name total is not defined' and aborted the tail
    # of the handler (the detector-blind-spot analysis never printed).
    total =sum (_class_counts .values ())
    print (f"[CLASS-HEALTH {tag }] detector emitted {total } objects across "
    f"{len (_class_counts )} class id(s):")
    for cid in sorted (_class_counts ):
        n =_class_counts [cid ]
        rej =_class_rejected .get (cid ,0 )
        name =LABELS [cid ]if 0 <=cid <len (LABELS )else "*** UNKNOWN ***"
        pct =100.0 *n /total
        note =""
        if not (0 <=cid <len (LABELS )):
            note ="  <-- NOT IN labels.txt: ALL DISCARDED, config mismatch"
        elif rej ==n and n >0 :
            if not is_person_class (cid ):
                note ="  (non-person: correctly ignored)"
            else :
                note =("  <-- 100% REJECTED but classified as a person class! "
                "check labels.txt person/non-person tags or PERSON_KEYWORDS")
        print (f"[CLASS-HEALTH {tag }]   {cid :>3}  {name :<26} "
        f"n={n :<7} ({pct :5.1f}%)  rejected={rej }{note }")
    if _unknown_cids :
        print (f"[CLASS-HEALTH {tag }] *** {len (_unknown_cids )} UNKNOWN CLASS ID(S) "
        f"{sorted (_unknown_cids )} *** labels.txt defines only "
        f"0..{len (LABELS )-1 }. Detections of these classes are being "
        "silently discarded. Fix labels.txt / num-detected-classes.")

    # DETECTOR BLIND SPOTS. A class that NEVER fires is a model/threshold matter,
    # not an app one. But raw "class never detected" is MISLEADING here: this
    # model always emits a POSTURE variant, so the postureless base classes
    # (e.g. 3 'Adult - Male', 9 'Kid - Girl') and any unseen posture are expected
    # to be empty whenever a sibling of the SAME demographic fired. Reporting
    # those as blind spots is noise. So split the never-detected classes into:
    #   (a) a DEMOGRAPHIC whose EVERY class is empty -> a genuine gap (that kind
    #       of person was never detected, or nobody of that kind was in view), and
    #   (b) a specific class empty while a SIBLING of the same demographic fired
    #       -> expected/redundant (the model expressed that person via another
    #       label), reported only as info.
    if total >0 :
        _fam_classes =defaultdict (list )   # family -> [cid,...]
        for cid in range (len (LABELS )):
            if not is_person_class (cid ):
                continue
            _demo ,_post =split_label (LABELS [cid ])
            _fam =demo_group (_demo )or (_demo or LABELS [cid ])
            _fam_classes [_fam ].append (cid )
        _dead =[]    # (family, [cids]) — no class in this demographic ever fired
        _redundant =[]   # (family, [empty cids]) — a sibling DID fire
        for _fam ,_cids in _fam_classes .items ():
            _fired =[c for c in _cids if _class_counts .get (c ,0 )>0 ]
            _empty =[c for c in _cids if _class_counts .get (c ,0 )==0 ]
            if not _empty :
                continue
            if _fired :
                _redundant .append ((_fam ,_empty ))
            else :
                _dead .append ((_fam ,_cids ))
        if _dead :
            print (f"[CLASS-HEALTH {tag }] *** {len (_dead )} DEMOGRAPHIC(S) NEVER "
            f"DETECTED in {total } objects (no class of this kind fired) ***")
            for _fam ,_cids in _dead :
                _names =", ".join (f"{c }:{LABELS [c ]}"for c in _cids )
                print (f"[CLASS-HEALTH {tag }]   {_fam }: {_names }")
            print (f"[CLASS-HEALTH {tag }]   If people of these types HAVE been in "
            "view, the detector is not finding them (MODEL/THRESHOLD limit — the "
            f"app rejected {sum (_class_rejected .values ())} objects in total). "
            "Try lowering minDetectorConfidence in config_tracker_reid.yml and "
            "pre-cluster-threshold in config_infer.txt, or retrain that class. "
            "If nobody of that type was on camera, this line is expected.")
        if _redundant :
            _tot_empty =sum (len (e )for _f ,e in _redundant )
            print (f"[CLASS-HEALTH {tag }] (info) {_tot_empty } class id(s) never "
            "fired but a SIBLING of the same demographic did — the model simply "
            "used a different posture/label, NOT a blind spot:")
            for _fam ,_empty in _redundant :
                _names =", ".join (f"{c }:{LABELS [c ]}"for c in _empty )
                print (f"[CLASS-HEALTH {tag }]   {_fam }: {_names }")

def camera_health_report (final =False ):
    tag ="FINAL"if final else "REPORT"
    cams =sorted (set (list (_cam_frames .keys ())+list (_cam_raw .keys ())))
    if not cams :
        print (f"[CAM-HEALTH {tag }] no frames from any camera yet.")
        return

    print (f"[CAM-HEALTH {tag }] per-camera pipeline:")
    if _src_idx_seen :
        per =", ".join (f"pad{n }={_src_idx_seen [n ]}"for n in sorted (_src_idx_seen ))
        print (f"[CAM-HEALTH {tag }]   frames seen per pad_index: {per }")
        if len (_src_idx_seen )==1 and len (camera_map )>1 :
            print (f"[CAM-HEALTH {tag }]   *** pad_index IS STILL COLLAPSED *** "
            f"{len (camera_map )} cameras configured but every frame reports "
            f"pad_index={next (iter (_src_idx_seen ))}. With nvurisrcbin this "
            "should not happen; check that each source linked to a distinct "
            "streammux sink pad.")
        else :
            print (f"[CAM-HEALTH {tag }]   per-camera attribution is WORKING "
            f"({len (_src_idx_seen )} distinct pad_index values seen)")
    if _src_tag_seen :
        per =", ".join (f"src{n }={_src_tag_seen [n ]}"for n in sorted (_src_tag_seen ))
        print (f"[CAM-HEALTH {tag }]   buffers leaving each source bin: {per }")
    dead_decode ,dead_detect =[],[]
    for c in cams :
        f ,r ,p =_cam_frames [c ],_cam_raw [c ],_cam_persons [c ]
        n_ids =len (_cam_ids [c ])
        print (f"[CAM-HEALTH {tag }]   {c :<14} frames={f :<7} raw={r :<7} "
        f"persons={p :<7} uids={n_ids }")
        if f ==0 :
            dead_decode .append (c )
        elif r ==0 :
            dead_detect .append (c )

    if dead_decode :
        print (f"[CAM-HEALTH {tag }] *** NO FRAMES from {', '.join (dead_decode )} *** "
        "the RTSP/decode/muxer path for these sources is not delivering. "
        "Cross-camera re-ID is IMPOSSIBLE while this is true.")
    n_configured =len (camera_map )
    n_reporting =len ([c for c in cams if _cam_frames [c ]>0 ])
    if n_configured >1 and n_reporting ==1 :
        print (f"[CAM-HEALTH {tag }] *** ALL {n_configured } CAMERAS ARE COLLAPSING "
        f"INTO ONE *** {n_configured } sources are configured but every "
        f"frame is attributed to '{cams [0 ]}'. Cross-camera hand-off cannot "
        "work: every track looks like it is on the same camera.")
    if dead_detect :
        print (f"[CAM-HEALTH {tag }] *** FRAMES BUT ZERO DETECTIONS from "
        f"{', '.join (dead_detect )} *** the muxer delivers frames for this "
        "source but the detector emits nothing for it. On a live RTSP setup "
        "this is almost always the nvstreammux batching race: a jittery source "
        "arrives late and misses every batch cycle, so its frames reach the "
        "probe (frame count rises) but carry no detected objects. Fixes: raise "
        "MUX_PUSH_TIMEOUT_US (now controls the muxer wait; default 160000us), "
        "confirm pgie batch-size >= number of sources, verify the RTSP link is "
        "stable (packet loss/reconnects), and check people are actually in view.")

    live =[c for c in cams if _cam_raw [c ]>0 ]
    multi ={}
    for c in cams :
        for pid in _cam_ids [c ]:
            multi .setdefault (pid ,set ()).add (c )
    handed ={p :cs for p ,cs in multi .items ()if len (cs )>1 }
    print (f"[CAM-HEALTH {tag }] cross-camera hand-off: "
    f"{len (live )} camera(s) producing detections, "
    f"{len (multi )} distinct UIDs, {len (handed )} seen on >1 camera")
    print (f"[CAM-HEALTH {tag }]   hand-offs succeeded: {_cam_handoffs ['ok']}   "
    f"possible misses: {_cam_handoffs ['missed']}")
    if len (live )>=2 and _cam_handoffs ["ok"]==0 and _cam_handoffs ["missed"]>3 :
        print (f"[CAM-HEALTH {tag }] *** HAND-OFF LOOKS BROKEN *** "
        f"{_cam_handoffs ['missed']} new UIDs were minted while another "
        "camera had live tracks, and NOT ONE hand-off succeeded. Check the "
        f"cross-camera bar ({CROSS_CAMERA_THRESHOLD :.2f}) against the "
        "cross-identity floor in the REID-HEALTH report above.")
    if len (live )<2 :
        print (f"[CAM-HEALTH {tag }] cross-camera hand-off CANNOT be evaluated: "
        "fewer than two cameras are producing detections.")
    elif not handed :
        print (f"[CAM-HEALTH {tag }] NO UID has been seen on more than one camera.")
    else :
        for p ,cs in list (handed .items ())[:8 ]:
            print (f"[CAM-HEALTH {tag }]   {p } seen on: {', '.join (sorted (cs ))}")

_last_cleanup =0.0
_reid_seen_once =False

_frame_count =0
_person_count =0
_reid_count =0
_raw_obj_count =0
_bad_box_count =0
_last_diag_time =0.0
_bad_box_logged =0
_good_box_logged =0
DIAG_EVERY =150

def l2_normalize (v ):
    v =np .asarray (v ,dtype =np .float32 )
    n =np .linalg .norm (v )
    return v /n if n >0 else v

def _l2_mean (vecs ):
    m =np .mean (np .asarray (vecs ,dtype =np .float32 ),axis =0 )
    return l2_normalize (m )

REID_HEALTH_ENABLE =True
REID_PROBE_SAMPLE =400
REID_TRACKS_TRACKED =24
REID_VECS_PER_TRACK =4
REID_REPORT_EVERY =1500

_reid_health ={
"checked":0 ,"ok":0 ,"none":0 ,"zero":0 ,"nan":0 ,"baddim":0 ,
"notunit":0 ,"constant":0 ,"db_writes":0 ,"db_rejects":0 ,
"track_samples":OrderedDict (),
"engine_ok":None ,"engine_msg":"",
}

def class_config_preflight (infer_cfg_path ="config_infer.txt"):

    print ("[CLASS-PREFLIGHT] verifying labels.txt <-> config_infer.txt agree")
    print (f"[CLASS-PREFLIGHT]   labels.txt: {len (LABELS )} classes loaded")
    try :
        with open (infer_cfg_path )as f :
            txt =f .read ()
    except Exception as e :
        print (f"[CLASS-PREFLIGHT] WARN — cannot read {infer_cfg_path }: {e }")
        return False

    ok =True
    m =re .search (r"^\s*num-detected-classes\s*=\s*(\d+)",txt ,re .M )
    if not m :
        print ("[CLASS-PREFLIGHT] WARN — num-detected-classes not found in config")
        ok =False
    else :
        n_cfg =int (m .group (1 ))
        print (f"[CLASS-PREFLIGHT]   config_infer.txt: num-detected-classes={n_cfg }")
        if n_cfg !=len (LABELS ):
            ok =False
            print (f"[CLASS-PREFLIGHT] *** MISMATCH *** the detector is configured "
            f"for {n_cfg } classes but labels.txt has {len (LABELS )}.")
            if n_cfg >len (LABELS ):
                missing =n_cfg -len (LABELS )
                print (f"[CLASS-PREFLIGHT]     The detector can emit class ids "
                f"{len (LABELS )}..{n_cfg -1 }. Those {missing } class(es) have "
                "NO label, so the app will SILENTLY DISCARD every one of "
                "their detections. Add the missing lines to labels.txt.")
            else :
                print (f"[CLASS-PREFLIGHT]     labels.txt lists classes the "
                f"detector can never emit (ids {n_cfg }..{len (LABELS )-1 }).")

    if re .search (r"^\s*labelfile-path\s*=",txt ,re .M ):
        print ("[CLASS-PREFLIGHT] WARN — labelfile-path is set in config_infer.txt. "
        "On the 9.1/pyservicemaker build it should be REMOVED so nvinfer "
        "does not draw object labels the app cannot override. See "
        "MIGRATION_AND_SETUP.md.")

    persons ,non_persons =[],[]
    for i ,name in enumerate (LABELS ):
        if is_person_class (i ):
            persons .append (f"{i }:{name }")
        else :
            non_persons .append (f"{i }:{name }")
    _mode =("ALL labeled classes (labels.txt is the filter)"if TRACK_ALL_LABELED
    else "person-keyword match")
    print (f"[CLASS-PREFLIGHT]   tracking mode: {_mode }")
    print (f"[CLASS-PREFLIGHT]   TRACKED ({len (persons )}): "
    f"{', '.join (persons )if persons else '<none>'}")
    print (f"[CLASS-PREFLIGHT]   excluded via !nonperson tag ({len (non_persons )}): "
    f"{', '.join (non_persons )if non_persons else '<none>'}")
    if not persons :
        ok =False
        print ("[CLASS-PREFLIGHT] *** NO CLASS IS TRACKABLE ***")

    print ("[CLASS-PREFLIGHT] PASS — class configuration is consistent"if ok
    else "[CLASS-PREFLIGHT] FAIL — fix the mismatches above; detections "
    "WILL be silently lost until you do")
    return ok

def reid_engine_preflight (tracker_cfg_path ):

    info ={}
    try :
        with open (tracker_cfg_path )as f :
            txt =f .read ()
    except Exception as e :
        _reid_health ["engine_ok"]=False
        _reid_health ["engine_msg"]=f"cannot read {tracker_cfg_path }: {e }"
        print (f"[REID-PREFLIGHT] FAIL — {_reid_health ['engine_msg']}")
        return False

    for key in ("modelEngineFile","onnxFile","reidFeatureSize","networkMode",
    "reidType","outputReidTensor"):
        m =re .search (rf"^\s*{key }\s*:\s*\"?([^\"#\n]+)\"?",txt ,re .M )
        if m :
            info [key ]=m .group (1 ).strip ()

    print ("[REID-PREFLIGHT] tracker config:",tracker_cfg_path )
    for k in ("reidType","outputReidTensor","reidFeatureSize","networkMode"):
        print (f"[REID-PREFLIGHT]   {k }: {info .get (k ,'<unset>')}")

    ok =True
    msgs =[]
    rt =info .get ("reidType")
    if rt not in ("1","2"):
        ok =False
        msgs .append (f"reidType={rt } — must be 1 (NvDeepSORT) or 2 for ReID output")
    if info .get ("outputReidTensor")not in ("1",None ):
        msgs .append ("outputReidTensor is not 1 — per-object embeddings may not be emitted")

    eng =info .get ("modelEngineFile")
    _onnx =info .get ("onnxFile")
    if not eng and not _onnx :
        ok =False
        msgs .append ("neither modelEngineFile nor onnxFile set in the ReID block")
    else :
        if _onnx :
            print (f"[REID-PREFLIGHT]   onnxFile: {_onnx }")
            if not os .path .isfile (_onnx ):
                ok =False
                msgs .append (f"onnxFile DOES NOT EXIST: {_onnx }")
        if eng :
            print (f"[REID-PREFLIGHT]   modelEngineFile: {eng }")
            if not os .path .isfile (eng ):
                if _onnx and os .path .isfile (_onnx ):
                    print ("[REID-PREFLIGHT]   engine not present yet — the tracker "
                    "will BUILD it from onnxFile on first run (takes a few minutes).")
                else :
                    ok =False
                    msgs .append (f"engine file DOES NOT EXIST: {eng }")
            else :
                sz =os .path .getsize (eng )
                print (f"[REID-PREFLIGHT]   engine size: {sz /1e6 :.1f} MB")
                if sz <1_000_000 :
                    ok =False
                    msgs .append (f"engine file suspiciously small ({sz } bytes)")

    try :
        fs =int (info .get ("reidFeatureSize",0 ))
        if fs and fs !=REID_FEATURE_SIZE :
            ok =False
            msgs .append (f"reidFeatureSize={fs } but the app/DB expect "
            f"{REID_FEATURE_SIZE } — vectors will mismatch")
    except ValueError :
        pass

    _reid_health ["engine_ok"]=ok
    _reid_health ["engine_msg"]="; ".join (msgs )
    if ok and not msgs :
        print ("[REID-PREFLIGHT] PASS — ReID engine present and config consistent")
    elif ok :
        for m in msgs :
            print (f"[REID-PREFLIGHT] WARN — {m }")
    else :
        for m in msgs :
            print (f"[REID-PREFLIGHT] FAIL — {m }")
        print ("[REID-PREFLIGHT] Re-identification WILL NOT WORK until this is "
        "fixed: every track will mint a new UID.")
    return ok

def reid_validate (vec ,expect =None ):
    expect =expect or REID_FEATURE_SIZE
    if vec is None :
        return False ,"none"
    v =np .asarray (vec ,dtype =np .float32 )
    if v .ndim !=1 or v .shape [0 ]!=expect :
        return False ,"baddim"
    if not np .all (np .isfinite (v )):
        return False ,"nan"
    n =float (np .linalg .norm (v ))
    if n <1e-6 :
        return False ,"zero"
    if abs (n -1.0 )>0.05 :
        return False ,"notunit"
    if float (np .std (v ))<1e-6 :
        return False ,"constant"
    return True ,"ok"

def reid_health_record (vec ,tkey =None ):
    if not REID_HEALTH_ENABLE :
        return True
    h =_reid_health
    h ["checked"]+=1
    ok ,why =reid_validate (vec )
    if ok :
        h ["ok"]+=1
        if tkey is not None :
            ts =h ["track_samples"]
            buf =ts .get (tkey )
            if buf is None :
                while len (ts )>=REID_TRACKS_TRACKED :
                    ts .popitem (last =False )
                buf =[]
                ts [tkey ]=buf
            if len (buf )<REID_VECS_PER_TRACK and h ["checked"]%30 ==0 :
                buf .append (np .asarray (vec ,dtype =np .float32 ))
    else :
        h [why ]=h .get (why ,0 )+1
        if h [why ]<=3 :
            print (f"[REID-HEALTH] BAD VECTOR ({why }) — this embedding cannot "
            "match anything; re-identification will fail for this track.")
    return ok

def reid_health_report (final =False ):
    if not REID_HEALTH_ENABLE :
        return
    h =_reid_health
    tag ="FINAL"if final else "REPORT"
    if h ["checked"]==0 :
        print (f"[REID-HEALTH {tag }] NO VECTORS SEEN AT ALL. The ReID engine is "
        "not producing embeddings — check reidType/outputReidTensor and "
        "the modelEngineFile path in the tracker config.")
        return

    good_pct =100.0 *h ["ok"]/max (h ["checked"],1 )
    print (f"[REID-HEALTH {tag }] vectors={h ['checked']} valid={h ['ok']} "
    f"({good_pct :.1f}%) zero={h ['zero']} nan={h ['nan']} "
    f"baddim={h ['baddim']} notunit={h ['notunit']} "
    f"constant={h ['constant']} none={h ['none']}")
    print (f"[REID-HEALTH {tag }] db_writes={h ['db_writes']} "
    f"db_rejects={h ['db_rejects']} (rejects are NORMAL: redundant or "
    "outlier crops are filtered on purpose)")

    ts =h ["track_samples"]
    tracks =[(k ,v )for k ,v in ts .items ()if len (v )>=1 ]
    if len (tracks )>=3 :
        cross ,within =[],[]
        for i ,(_ ,vi )in enumerate (tracks ):
            Mi =np .asarray (vi ,dtype =np .float32 )
            if len (vi )>=2 :
                s =Mi @Mi .T
                iu =np .triu_indices (len (vi ),k =1 )
                within .extend (s [iu ].tolist ())
            for j in range (i +1 ,len (tracks )):
                Mj =np .asarray (tracks [j ][1 ],dtype =np .float32 )
                cross .extend ((Mi @Mj .T ).ravel ().tolist ())
        if cross :
            c_mean =float (np .mean (cross ))
            c_p95 =float (np .percentile (cross ,95 ))
            w_mean =float (np .mean (within ))if within else float ("nan")
            margin =(w_mean -c_mean )if within else float ("nan")
            print (f"[REID-HEALTH {tag }] discriminativeness over {len (tracks )} "
            f"distinct identities:")
            print (f"[REID-HEALTH {tag }]   cross-identity (different people): "
            f"mean={c_mean :.3f} p95={c_p95 :.3f}   <- the stranger floor")
            if within :
                print (f"[REID-HEALTH {tag }]   within-identity (same person)  : "
                f"mean={w_mean :.3f}")
                print (f"[REID-HEALTH {tag }]   margin (same - different)      : "
                f"{margin :.3f}")
            print (f"[REID-HEALTH {tag }]   thresholds in use: same-cam "
            f"{SIMILARITY_THRESHOLD :.2f}  cross-cam "
            f"{CROSS_CAMERA_THRESHOLD :.2f}")
            if c_mean >0.90 :
                print (f"[REID-HEALTH {tag }] *** DEGENERATE FEATURES *** even "
                "DIFFERENT tracks score >0.90 against each other. Check "
                "inferDims / offsets / netScaleFactor / colorFormat.")
            elif CROSS_CAMERA_THRESHOLD <=c_p95 :
                print (f"[REID-HEALTH {tag }] NOTE: the cross-camera bar "
                f"({CROSS_CAMERA_THRESHOLD :.2f}) sits at/below the p95 of "
                f"the stranger distribution ({c_p95 :.3f}).")
            elif within and margin <0.15 :
                print (f"[REID-HEALTH {tag }] NOTE: only {margin :.3f} separates "
                "the same person from a stranger.")

    if good_pct <50.0 :
        print (f"[REID-HEALTH {tag }] *** MORE THAN HALF OF VECTORS ARE INVALID ***")
    if h ["engine_ok"]is False :
        print (f"[REID-HEALTH {tag }] preflight had FAILED: {h ['engine_msg']}")

def _l2_mean_checked (vecs ):
    m =_l2_mean (vecs )
    ok ,why =reid_validate (m )
    if not ok :
        print (f"[REID-HEALTH] refusing to commit identity from a bad centroid "
        f"({why }) — {len (vecs )} vectors averaged to nothing usable.")
        return None
    return m

def maybe_cleanup ():
    global _last_cleanup
    now =time .time ()
    if now -_last_cleanup <CLEANUP_INTERVAL :
        return
    _last_cleanup =now
    for key in [k for k ,ts in tracker_last_seen .items ()
    if now -ts >TRACKER_TIMEOUT ]:
        tracker_last_seen .pop (key ,None )
        tracker_present_ts .pop (key ,None )
        tracker_to_person .pop (key ,None )
        _track_emb_buffer .pop (key ,None )
        tracker_class_votes .pop (key ,None )
        tracker_posture_recent .pop (key ,None )
        tracker_hits .pop (key ,None )
        tracker_seen .pop (key ,None )
        tracker_positions .pop (key ,None )
        tracker_is_static .pop (key ,None )
        tracker_ever_moved .pop (key ,None )
        tracker_first_seen_ts .pop (key ,None )
        _track_first_center .pop (key ,None )
        _track_last_center .pop (key ,None )
        _track_peak_disp .pop (key ,None )
        tracker_motion_frames .pop (key ,None )
        _track_motion_run .pop (key ,None )
        _track_center_win .pop (key ,None )
        tracker_size_hist .pop (key ,None )
        tracker_conf_hist .pop (key ,None )
        _appearance_confirmed .discard (key )
        _person_streak .pop (key ,None )
        _deform_streak .pop (key ,None )
        tracker_deform .pop (key ,None )
        _last_gallery_update .pop (key ,None )
        _last_gallery_update .pop (("merge",key ),None )
        _reid_health ["track_samples"].pop (key ,None )
    # Prune stale spatial-recovery entries (older than the recovery window).
    _now =time .time ()
    # Enforce the retention policy on the MCMTT gallery: identities unseen for
    # RETENTION_SECONDS (30 min) are dropped, so re-identification works within
    # the window and memory stays bounded beyond it.
    if GALLERY is not None :
        try :
            _gone =GALLERY .prune (max_age =RETENTION_SECONDS ,now =_now )
            if _gone :
                print (f"[MCMTT] pruned {_gone } identities unseen for "
                f">{RETENTION_SECONDS /60 :.0f} min "
                f"({len (GALLERY )} retained)")
        except Exception as _e :
            print (f"[MCMTT] prune failed: {_e }")

    for _si in list (recent_local_tracks .keys ()):
        recent_local_tracks [_si ]=[e for e in recent_local_tracks [_si ]
        if _now -e [3 ]<=RECOVER_MAX_GAP ]
        if not recent_local_tracks [_si ]:
            recent_local_tracks .pop (_si ,None )

_reid_api_logged =False
_reid_diag_done =False
_reid_obj_introspected =False
_batch_diag_done =False

# Holds the per-batch ReID tensor for the batch currently being processed, so
# per-object index-style metadata can be resolved into a feature vector.
# Populated at the top of _handle_batch, read inside extract_reid.
_batch_reid_features =None      # np.ndarray [numFilled, featureSize] or None
_batch_reid_map =None           # dict object_id -> reid vector (via pyds), or None
_batch_reid_logged =False


def _as_float_array (obj ,size =None ):
    """Best-effort conversion of a pyservicemaker feature handle to np.float32.
    Handles python lists/tuples, numpy arrays, memoryview / buffer-protocol
    objects, ctypes float arrays, and DLPack/torch-like tensors."""
    if obj is None :
        return None
    arr =None
    # 1) direct numpy / list / tuple
    try :
        arr =np .array (obj ,dtype =np .float32 ,copy =True )
        if arr .ndim ==0 :
            arr =None
    except Exception :
        arr =None
    # 2) buffer protocol / memoryview
    if arr is None :
        try :
            arr =np .frombuffer (memoryview (obj ),dtype =np .float32 ).copy ()
        except Exception :
            arr =None
    # 3) DLPack / torch-like
    if arr is None :
        try :
            import numpy as _np2  # noqa: PLC0415
            if hasattr (obj ,"__dlpack__"):
                arr =_np2 .from_dlpack (obj ).astype (np .float32 ).reshape (-1 ).copy ()
            elif hasattr (obj ,"numpy"):
                arr =obj .numpy ().astype (np .float32 ).reshape (-1 ).copy ()
        except Exception :
            arr =None
    # 4) generic iterable of floats
    if arr is None :
        try :
            arr =np .array (list (obj ),dtype =np .float32 )
        except Exception :
            return None
    if arr is None :
        return None
    arr =arr .reshape (-1 )
    if size and arr .size >=size :
        arr =arr [:size ]
    return arr


_ct =ctypes


class _ReidTensorC (_ct .Structure ):
    """Memory layout of NvDsReidTensorBatch (nvds_tracker_meta.h, DS 7.1/9.x).
    Proven correct on-device: reading ptr_host yields L2-normalized 256-d
    vectors (norm=1.0000). We read it directly because this pyservicemaker /
    pyds build lacks the NvDsReidTensorBatch Python binding."""
    _fields_ =[
    ("featureSize",_ct .c_uint32 ),
    ("numFilled",_ct .c_uint32 ),
    ("ptr_host",_ct .c_void_p ),
    ("ptr_dev",_ct .c_void_p ),
    ("priv_data",_ct .c_void_p ),
    ]


_reid_ptr_accessor =None      # cached name of the accessor that yields the raw addr
_reid_introspected =False


def _meta_raw_address (user_meta ):
    """Return the integer address of a pyservicemaker/pyds user-meta payload,
    trying the known accessors. Caches the one that works."""
    global _reid_ptr_accessor
    # Fast path: use the cached accessor.
    if _reid_ptr_accessor is not None :
        try :
            f =getattr (user_meta ,_reid_ptr_accessor ,None )
            val =f ()if callable (f )else f
            if isinstance (val ,int )and val >0x1000 :
                return val
        except Exception :
            pass
    # Discovery: try likely accessors for the raw pointer / payload address.
    for acc in ("raw_ptr","raw_address","address","ptr","get_ptr",
    "user_meta_data","data","user_data","payload_ptr"):
        f =getattr (user_meta ,acc ,None )
        if f is None :
            continue
        try :
            val =f ()if callable (f )else f
        except Exception :
            continue
        if isinstance (val ,int )and val >0x1000 :
            _reid_ptr_accessor =acc
            return val
    # pyds fallback: some builds expose a module-level get_ptr(user_meta_data).
    try :
        import pyds  # noqa: PLC0415
        data =getattr (user_meta ,"user_meta_data",None )
        if data is None :
            data =getattr (user_meta ,"data",None )
        if data is not None :
            addr =pyds .get_ptr (data )
            if isinstance (addr ,int )and addr >0x1000 :
                _reid_ptr_accessor ="__pyds_get_ptr__"
                return addr
    except Exception :
        pass
    return None


def _introspect_reid_meta (user_meta ):
    """One-shot dump of the ReID user-meta API surface, so the pointer accessor
    can be identified if the defaults above miss it."""
    global _reid_introspected
    if _reid_introspected :
        return
    _reid_introspected =True
    try :
        names =[a for a in dir (user_meta )if not a .startswith ("__")]
        print (f"[REID-INTROSPECT] UserMetadata type={type (user_meta )}")
        print (f"[REID-INTROSPECT] attrs/methods: {names }")
        for acc in ("user_meta_data","data","get_ptr","ptr","raw_ptr",
        "address","as_reid_tensor","as_obj_reid"):
            f =getattr (user_meta ,acc ,None )
            if f is None :
                continue
            try :
                val =f ()if callable (f )else f
                print (f"[REID-INTROSPECT]   {acc } -> {type (val )}  "
                f"{str (val )[:60 ]}")
            except Exception as e :
                print (f"[REID-INTROSPECT]   {acc } raised: {e }")
    except Exception as e :
        print (f"[REID-INTROSPECT] failed: {e }")


_pyds_mod =None
_pyds_batch_accessor =None
_pyds_logged =False


def _get_pyds ():
    # DISABLED: raw pyds pointer walking segfaulted on this build. ReID is read
    # natively via obj_reid_items().as_obj_reid().featureVector() instead. This
    # stub guarantees the crashing path can never execute even if called.
    return None


def _batch_meta_address (batch_meta ):
    """Find the raw NvDsBatchMeta* address behind a pyservicemaker batch_meta."""
    global _pyds_batch_accessor
    if _pyds_batch_accessor is not None :
        try :
            f =getattr (batch_meta ,_pyds_batch_accessor ,None )
            v =f ()if callable (f )else f
            if isinstance (v ,int )and v >0x1000 :
                return v
        except Exception :
            pass
    for acc in ("ptr","raw_ptr","address","get_ptr","native_ptr","c_ptr",
    "handle","meta_ptr"):
        f =getattr (batch_meta ,acc ,None )
        if f is None :
            continue
        try :
            v =f ()if callable (f )else f
        except Exception :
            continue
        if isinstance (v ,int )and v >0x1000 :
            _pyds_batch_accessor =acc
            return v
    return None


def _load_batch_reid_via_pyds (batch_meta ):
    """Read the batch ReID tensor using classic pyds bindings (proven to see the
    meta on this device). Returns [numFilled, featureSize] float32 or None."""
    global _batch_reid_logged ,_pyds_logged
    pyds =_get_pyds ()
    if pyds is None :
        if not _pyds_logged :
            print ("[REID-API] pyds not importable for fallback ReID read.")
            _pyds_logged =True
        return None
    addr =_batch_meta_address (batch_meta )
    if addr is None :
        if not _pyds_logged :
            print ("[REID-INTROSPECT] could not get raw batch-meta pointer from "
            f"pyservicemaker; batch_meta attrs: "
            f"{[a for a in dir (batch_meta )if not a .startswith ('__')]}")
            _pyds_logged =True
        return None
    try :
        bm =pyds .NvDsBatchMeta .cast (addr )
    except Exception as e :
        if not _pyds_logged :
            print (f"[REID-API] pyds NvDsBatchMeta.cast failed: {e }")
            _pyds_logged =True
        return None
    try :
        l =bm .batch_user_meta_list
        while l is not None :
            try :
                um =pyds .NvDsUserMeta .cast (l .data )
            except StopIteration :
                break
            if um and um .base_meta .meta_type ==NVDS_TRACKER_BATCH_REID_META :
                daddr =pyds .get_ptr (um .user_meta_data )
                t =_ct .cast (daddr ,_ct .POINTER (_ReidTensorC )).contents
                fsz =int (t .featureSize );nf =int (t .numFilled )
                if t .ptr_host and nf >0 and fsz >0 :
                    buf =(_ct .c_float *(nf *fsz )).from_address (t .ptr_host )
                    arr =np .ctypeslib .as_array (buf ).reshape (nf ,fsz ).copy ()
                    if not _batch_reid_logged :
                        print (f"[REID-API] batch ReID tensor read via PYDS "
                        f"fallback: shape={arr .shape }")
                        _batch_reid_logged =True
                    return arr
                return None
            try :
                l =l .next
            except StopIteration :
                break
    except Exception as e :
        if not _pyds_logged :
            print (f"[REID-API] pyds batch walk failed: {e }")
            _pyds_logged =True
    return None


def _build_reid_map_via_pyds (batch_meta ,batch_feats ):
    """Walk the batch with pyds and map object_id -> reid vector, using the
    per-object NVDS_TRACKER_OBJ_REID_META index into batch_feats. Returns a dict
    or None. This avoids cross-API object matching: the pyservicemaker side just
    looks up obj.object_id in this dict."""
    pyds =_get_pyds ()
    if pyds is None or batch_feats is None :
        return None
    addr =_batch_meta_address (batch_meta )
    if addr is None :
        return None
    try :
        bm =pyds .NvDsBatchMeta .cast (addr )
    except Exception :
        return None
    out ={}
    try :
        lf =bm .frame_meta_list
        while lf is not None :
            try :
                fm =pyds .NvDsFrameMeta .cast (lf .data )
            except StopIteration :
                break
            lo =fm .obj_meta_list
            while lo is not None :
                try :
                    om =pyds .NvDsObjectMeta .cast (lo .data )
                except StopIteration :
                    break
                lu =om .obj_user_meta_list
                while lu is not None :
                    try :
                        um =pyds .NvDsUserMeta .cast (lu .data )
                    except StopIteration :
                        break
                    if um and um .base_meta .meta_type ==NVDS_TRACKER_OBJ_REID_META :
                        try :
                            ip =_ct .cast (pyds .get_ptr (um .user_meta_data ),
                            _ct .POINTER (_ct .c_int32 ))
                            idx =int (ip .contents .value )
                            if 0 <=idx <batch_feats .shape [0 ]:
                                out [int (om .object_id )]=batch_feats [idx ]
                        except Exception :
                            pass
                    try :
                        lu =lu .next
                    except StopIteration :
                        break
                try :
                    lo =lo .next
                except StopIteration :
                    break
            try :
                lf =lf .next
            except StopIteration :
                break
    except Exception :
        return out or None
    return out or None


def load_batch_reid_tensor (batch_meta ):
    """Extract the whole-batch ReID tensor (NVDS_TRACKER_BATCH_REID_META) by
    reading the NvDsReidTensorBatch struct directly via ctypes. Returns a 2D
    np.float32 array [numFilled, featureSize] or None.

    Verified on-device: the tracker emits this every batch and the vectors are
    L2-normalized 256-d. We bypass the missing NvDsReidTensorBatch binding.
    """
    global _batch_reid_logged
    try :
        _umi =getattr (batch_meta ,"user_meta_items",None )
        _items =[]
        if callable (_umi ):
            try :
                _items =list (_umi (NVDS_TRACKER_BATCH_REID_META )or [])
            except TypeError :
                _items =[u for u in (_umi ()or [])
                if getattr (u ,"meta_type",None )==NVDS_TRACKER_BATCH_REID_META ]
        for u in _items :
            addr =_meta_raw_address (u )
            if addr is None :
                _introspect_reid_meta (u )
                continue
            try :
                t =_ct .cast (addr ,_ct .POINTER (_ReidTensorC )).contents
                fsz =int (t .featureSize )
                nf =int (t .numFilled )
                if not t .ptr_host or nf <=0 or fsz <=0 :
                    return None            # no vectors filled this batch
                buf =(_ct .c_float *(nf *fsz )).from_address (t .ptr_host )
                arr =np .ctypeslib .as_array (buf ).reshape (nf ,fsz ).copy ()
                if not _batch_reid_logged :
                    print (f"[REID-API] batch ReID tensor read via ctypes: "
                    f"shape={arr .shape } (accessor={_reid_ptr_accessor })")
                    _batch_reid_logged =True
                return arr
            except Exception as e :
                if not _batch_reid_logged :
                    print (f"[REID-API] ctypes tensor read failed: {e }")
                    _batch_reid_logged =True
                return None

        # ---- pyds fallback -------------------------------------------------
        # pyservicemaker's user_meta_items does NOT surface tracker ReID meta on
        # this build (confirmed: count=0 while a pyds probe saw it on every
        # batch). If we have a raw NvDsBatchMeta pointer we can walk it with pyds
        # exactly like the working probe did.
        arr =_load_batch_reid_via_pyds (batch_meta )
        if arr is not None :
            return arr
    except Exception as e :
        if not _batch_reid_logged :
            print (f"[REID-API] batch ReID tensor read failed: {e }")
            _batch_reid_logged =True
    return None

def _obj_user_meta_items (obj_meta ,meta_type ):
    """Iterate an object's user-meta items of a given type, tolerating both the
    filtered signature user_meta_items(meta_type) and the unfiltered one."""
    _acc =getattr (obj_meta ,"user_meta_items",None )
    if not callable (_acc ):
        return []
    try :
        return list (_acc (meta_type )or [])
    except TypeError :
        return [u for u in (_acc ()or [])
        if getattr (u ,"meta_type",None )==meta_type ]


def extract_reid (obj_meta ):
    """Return an L2-normalised ReID vector for this object, or None.

    Native pyservicemaker path (confirmed on-device):
        for rmeta in obj_meta.obj_reid_items():
            reid = rmeta.as_obj_reid()          -> ObjectReidUserMetadata
            size = reid.featureSize()           -> 256
            vec  = reid.featureVector()         -> the CPU feature vector
    No pyds, no ctypes, no raw pointers (those segfaulted on this build).
    """
    global _reid_seen_once ,_reid_api_logged ,_reid_obj_introspected
    try :
        _rit =getattr (obj_meta ,"obj_reid_items",None )
        if callable (_rit ):
            try :
                _rit =_rit ()
            except Exception :
                _rit =None
        for rmeta in (_rit or []):
            reid =None
            _f =getattr (rmeta ,"as_obj_reid",None )
            if callable (_f ):
                try :
                    reid =_f ()
                except Exception :
                    reid =None
            if reid is None :
                continue

            # size: featureSize() (method) or feature_size (attr)
            size =0
            for _sacc in ("featureSize","feature_size"):
                _s =getattr (reid ,_sacc ,None )
                try :
                    size =int (_s ()if callable (_s )else _s )
                except Exception :
                    size =0
                if size :
                    break
            if not size :
                size =REID_FEATURE_SIZE

            # vector: featureVector() (method) or feature_vector (attr)
            fv =None
            for _vacc in ("featureVector","feature_vector"):
                _v =getattr (reid ,_vacc ,None )
                if _v is None :
                    continue
                try :
                    fv =_v ()if callable (_v )else _v
                except Exception :
                    fv =None
                if fv is not None :
                    break

            # One-shot introspection if we still can't turn fv into an array.
            feat =_as_float_array (fv ,size )
            if (feat is None or feat .size <size )and not _reid_obj_introspected :
                _reid_obj_introspected =True
                try :
                    print (f"[REID-INTROSPECT] ObjectReidUserMetadata attrs: "
                    f"{[a for a in dir (reid )if not a .startswith ('__')]}")
                    print (f"[REID-INTROSPECT] featureVector() type={type (fv )} "
                    f"repr={str (fv )[:80 ]}")
                except Exception :
                    pass

            if feat is not None and feat .size >=size :
                if not _reid_api_logged :
                    print (f"[REID-API] using obj_reid_items.as_obj_reid()."
                    f"featureVector() (size={feat .size })")
                    _reid_api_logged =True
                _reid_seen_once =True
                return l2_normalize (feat [:size ])
    except Exception as e :
        if not _reid_api_logged :
            print (f"[REID-API] native obj_reid read failed: {e }")
            _reid_api_logged =True
    return None

def _iou (a ,b ):
    ax1 ,ay1 ,aw ,ah =a ;ax2 ,ay2 =ax1 +aw ,ay1 +ah
    bx1 ,by1 ,bw ,bh =b ;bx2 ,by2 =bx1 +bw ,by1 +bh
    ix1 ,iy1 =max (ax1 ,bx1 ),max (ay1 ,by1 )
    ix2 ,iy2 =min (ax2 ,bx2 ),min (ay2 ,by2 )
    iw ,ih =max (0.0 ,ix2 -ix1 ),max (0.0 ,iy2 -iy1 )
    inter =iw *ih
    if inter <=0 :
        return 0.0
    union =aw *ah +bw *bh -inter
    return inter /union if union >0 else 0.0

def is_static_box (tkey ,cx ,cy ,now ):
    if not STATIC_FLAG_ENABLE :
        return False
    hist =tracker_positions [tkey ]
    hist .append ((now ,cx ,cy ))
    window =[(t ,x ,y )for (t ,x ,y )in hist if now -t <=STATIC_MIN_SECONDS ]
    if len (window )<STATIC_MIN_SAMPLES :
        return False
    span =window [-1 ][0 ]-window [0 ][0 ]
    if span <STATIC_MIN_SECONDS *0.75 :
        return False
    xs =[x for _ ,x ,_ in window ]
    ys =[y for _ ,_ ,y in window ]
    mx =sum (xs )/len (xs )
    my =sum (ys )/len (ys )
    max_drift =max (((x -mx )**2 +(y -my )**2 )**0.5 for x ,y in zip (xs ,ys ))
    return max_drift <=STATIC_MAX_DRIFT_PX

def suppress_overlaps (frame_meta ,iou_thresh =0.5 ):

    boxes =[]
    for o in frame_meta .object_items :
        try :
            r =o .rect_params
            boxes .append ((o .object_id ,float (o .confidence ),
            (float (r .left ),float (r .top ),float (r .width ),float (r .height ))))
        except Exception :
            pass
    boxes .sort (key =lambda b :b [1 ],reverse =True )
    kept ,suppressed =[],set ()
    for oid ,conf ,box in boxes :
        if any (_iou (box ,kbox )>iou_thresh for _ ,kbox in kept ):
            suppressed .add (oid )
        else :
            kept .append ((oid ,box ))
    return suppressed

def expand_box (rect_params ,frame_w =FRAME_W ,frame_h =FRAME_H ):

    r =rect_params
    bw ,bh =int (r .width ),int (r .height )
    pad_x =int (bw *0.05 )
    pad_y =int (bh *0.05 )
    left =max (0 ,int (r .left )-pad_x )
    top =max (0 ,int (r .top )-pad_y )
    right =min (frame_w ,int (r .left )+bw +pad_x )
    bottom =min (frame_h ,int (r .top )+bh +pad_y )
    return left ,top ,right -left ,bottom -top

LOG_ROOT ="logs"
ROTATE_SEC =7200
log_file =writer =None
log_start_time =time .time ()
WINDOW_START =None
_log_lock =threading .Lock ()

# CSV summary schema (v23). ONE row per (camera, UID) carrying, for the identity
# in front of that camera: the demographic flags, the posture episode counts, and
# the TOTAL time that identity spent standing vs sitting on that camera.
#   Camera Name        - the camera the row is for
#   UID                - the committed person id (pid)
#   UID First Seen     - global first-seen datetime for this UID (from person DB)
#   UID Start Time     - first time this UID was seen ON THIS CAMERA this window
#   UID End Time       - last  time this UID was seen ON THIS CAMERA this window
#   isMale/isFemale/isBoy/isGirl - 1/0 flags from the pooled per-UID demographic
#   #Standing/#Sitting - number of distinct standing/sitting episodes observed
#   Total Standing/Sitting Time - accumulated seconds spent in each posture
CSV_HEADER =[
"Camera Name","UID","UID First Seen","UID Start Time","UID End Time",
"isMale","isFemale","isBoy","isGirl",
"#Standing","#Sitting","Total Standing Time","Total Sitting Time",
]

# Cap on the time gap (seconds) a single detection-to-detection step may add to a
# posture's running total. If a track disappears and the same (cam, UID) session
# reappears after a long occlusion, the elapsed interval is attributed to the
# last-known posture; this cap stops one long gap from dumping minutes into a
# single posture bucket. Env-tunable for on-device tuning.
POSTURE_STEP_CAP =float (os .environ .get ("CSV_POSTURE_STEP_CAP","5.0"))

# Per-(camera, UID) sessions accumulated over the current log window. Epoch
# seconds are stored so dwell is a simple subtraction; datetimes are formatted
# only at flush time. Guarded by an RLock (reentrant so the merge repointer can
# fold sessions while other session code is on the stack without deadlocking).
_sessions ={}                    # (cam, pid) -> session dict (see _touch_session)
_uid_cameras =defaultdict (set ) # pid -> {cam,...}  (window-global, for Camera List)
_sessions_lock =threading .RLock ()
_flush_frame_counter =[0 ]
FLUSH_EVERY =int (os .environ .get ("CSV_FLUSH_EVERY","50"))  # ~2s @26fps

def _norm_posture (posture ):
    """Collapse a posture label to 'standing' | 'sitting' | '' (unknown)."""
    if not posture :
        return ""
    p =posture .lower ()
    if "stand"in p :
        return "standing"
    if "sit"in p :
        return "sitting"
    return ""

def _touch_session (cam ,pid ,demo ,posture ,now_ts ):
    """Record/extend the dwell of committed UID `pid` on camera `cam`.
    Called once per committed detection per frame.

    Beyond the first/last timestamps this now tracks posture:
      * stand_time / sit_time  - seconds accumulated in each posture. Each frame
        the interval since the previous touch is added to whichever posture the
        person was in DURING that interval (capped by POSTURE_STEP_CAP so a long
        occlusion gap can't dump minutes into one bucket).
      * stand_count / sit_count - number of distinct posture EPISODES, i.e. how
        many separate times the person entered standing / sitting. The count is
        bumped on the first observation and on every transition into that posture.
    An empty/unknown posture neither accumulates time nor starts an episode; the
    previous known posture is carried forward until a clear posture is observed.
    """
    posture =_norm_posture (posture )
    with _sessions_lock :
        key =(cam ,pid )
        s =_sessions .get (key )
        if s is None :
            s ={"first":now_ts ,"last":now_ts ,"demo":demo or "",
            "posture":posture ,"stand_time":0.0 ,"sit_time":0.0 ,
            "stand_count":0 ,"sit_count":0 }
            if posture =="standing":
                s ["stand_count"]=1
            elif posture =="sitting":
                s ["sit_count"]=1
            _sessions [key ]=s
        else :
            # Attribute the elapsed interval to the posture held until now.
            dt =now_ts -s ["last"]
            if dt >0 :
                dt =min (dt ,POSTURE_STEP_CAP )
                if s ["posture"]=="standing":
                    s ["stand_time"]+=dt
                elif s ["posture"]=="sitting":
                    s ["sit_time"]+=dt
            s ["last"]=now_ts
            if demo and not s ["demo"]:
                s ["demo"]=demo
            # Posture transition -> start a new episode of the entered posture.
            if posture and posture !=s ["posture"]:
                if posture =="standing":
                    s ["stand_count"]+=1
                elif posture =="sitting":
                    s ["sit_count"]+=1
                s ["posture"]=posture
            elif posture and s ["posture"]=="":
                # First clear posture after an unknown start.
                if posture =="standing"and s ["stand_count"]==0 :
                    s ["stand_count"]=1
                elif posture =="sitting"and s ["sit_count"]==0 :
                    s ["sit_count"]=1
                s ["posture"]=posture
        _uid_cameras [pid ].add (cam )

def _repoint_sessions (drop_pid ,keep_pid ):
    """After a cross-camera merge, fold the dropped id's sessions into the
    survivor so the CSV shows one identity with one dwell per camera."""
    with _sessions_lock :
        for cam in [c for (c ,p )in list (_sessions .keys ())if p ==drop_pid ]:
            src =_sessions .pop ((cam ,drop_pid ),None )
            if src is None :
                continue
            dst =_sessions .get ((cam ,keep_pid ))
            if dst is None :
                _sessions [(cam ,keep_pid )]=src
            else :
                dst ["first"]=min (dst ["first"],src ["first"])
                dst ["last"]=max (dst ["last"],src ["last"])
                if src ["demo"]and not dst ["demo"]:
                    dst ["demo"]=src ["demo"]
                # Fold posture totals/episode counts from the dropped session.
                dst ["stand_time"]=dst .get ("stand_time",0.0 )+src .get ("stand_time",0.0 )
                dst ["sit_time"]=dst .get ("sit_time",0.0 )+src .get ("sit_time",0.0 )
                dst ["stand_count"]=dst .get ("stand_count",0 )+src .get ("stand_count",0 )
                dst ["sit_count"]=dst .get ("sit_count",0 )+src .get ("sit_count",0 )
                if not dst .get ("posture")and src .get ("posture"):
                    dst ["posture"]=src ["posture"]
        drop_cams =_uid_cameras .pop (drop_pid ,None )
        if drop_cams :
            _uid_cameras [keep_pid ]|=drop_cams

def _dominant_demo (pid ,fallback =""):
    """The pooled per-UID demographic (one identity = one demographic), resolved
    on independent age/gender axes so the CSV flags match the on-screen label."""
    return resolve_uid_demo (pid ,fallback )

def _demo_flags (demo ):
    """Map a demographic label to the four mutually-exclusive CSV flags
    (isMale, isFemale, isBoy, isGirl) as 1/0 integers. Unknown -> all zero."""
    g =demo_group (demo )
    return (
    1 if g =="Male"else 0 ,
    1 if g =="Female"else 0 ,
    1 if g =="Kid - Boy"else 0 ,
    1 if g =="Kid - Girl"else 0 ,
    )

def flush_sessions ():
    """Rewrite the whole CSV from the session table: header + one row per
    (camera, UID). Cheap even at 50-frame cadence because the table holds a few
    dozen identities, not tens of thousands of frame rows."""
    if writer is None or log_file is None :
        return
    with _sessions_lock :
        rows =[]
        for (cam ,pid ),s in _sessions .items ():
            first_ts ,last_ts =s ["first"],s ["last"]
            first_dt =datetime .fromtimestamp (first_ts )
            last_dt =datetime .fromtimestamp (last_ts )
            # Global first-seen for this UID (across the whole DB / all cameras).
            uid_first =""
            if DB is not None and not pid .startswith ("T"):
                try :
                    _ ,uid_first =DB .first_seen_of (pid )
                except Exception :
                    uid_first =""
            # Demographic flags from the pooled one-identity-one-demographic vote,
            # falling back to whatever this session last observed.
            demo =_dominant_demo (pid ,s .get ("demo",""))
            is_male ,is_female ,is_boy ,is_girl =_demo_flags (demo )
            stand_time =int (round (s .get ("stand_time",0.0 )))
            sit_time =int (round (s .get ("sit_time",0.0 )))
            rows .append ([
            cam ,pid ,uid_first ,
            first_dt .strftime ("%Y-%m-%d %H:%M:%S"),
            last_dt .strftime ("%Y-%m-%d %H:%M:%S"),
            is_male ,is_female ,is_boy ,is_girl ,
            s .get ("stand_count",0 ),s .get ("sit_count",0 ),
            stand_time ,sit_time ,
            ])
        # Deterministic order: first appearance (UID Start Time), then camera.
        rows .sort (key =lambda r :(r [3 ],r [0 ]))
        try :
            log_file .seek (0 )
            log_file .truncate ()
            writer .writerow (CSV_HEADER )
            for r in rows :
                writer .writerow (r )
            log_file .flush ()
        except Exception as e :
            print (f"[LOG] flush_sessions error: {e }")

def open_new_log ():
    global log_start_time ,WINDOW_START
    log_start_time =time .time ()
    WINDOW_START =datetime .now ()
    folder =os .path .join (LOG_ROOT ,WINDOW_START .strftime ("%Y-%m-%d"))
    os .makedirs (folder ,exist_ok =True )
    base =WINDOW_START .strftime ("%H-%M-%S")
    path =os .path .join (folder ,base +".csv")
    suffix =1
    while os .path .exists (path ):
        path =os .path .join (folder ,f"{base }_{suffix }.csv")
        suffix +=1
    f =open (path ,"w",newline ="")
    w =csv .writer (f )
    w .writerow (CSV_HEADER )
    f .flush ()
    # A rotation starts a fresh window: previous sessions belong to the old file.
    with _sessions_lock :
        _sessions .clear ()
        _uid_cameras .clear ()
        _flush_frame_counter [0 ]=0
    print (f"[LOG] opened {path } (rotates in {ROTATE_SEC //3600 }h "
    f"{(ROTATE_SEC %3600 )//60 }m)")
    return f ,w

def maybe_rotate ():
    global log_file ,writer
    if time .time ()-log_start_time <=ROTATE_SEC :
        return False
    with _log_lock :
        if time .time ()-log_start_time <=ROTATE_SEC :
            return False
        try :
            flush_sessions ()          # persist the closing window in full
            log_file .flush ()
            log_file .close ()
        except Exception as e :
            print (f"[LOG] close-on-rotate error: {e }")
        log_file ,writer =open_new_log ()
    return True

def _box_center (box ):
    l ,t ,w ,h =box
    return (l +w /2.0 ,t +h /2.0 )


def _committed_uid_overlapping_box (src_idx ,box ,now ,exclude_tkey =None ,
iou_thr =0.45 ,hold =None ):
    """Return a committed UID currently held by a LIVE track on this camera whose
    latest box overlaps `box` by >= iou_thr, or None. Used to stop a re-minted
    tracker object_id from minting a SECOND UID on a person who already has one
    (the TTE69K/HG138V same-camera split). Unlike _recover_local_uid — which
    inherits INACTIVE lost UIDs and deliberately skips active ones — this targets
    ACTIVE overlapping tracks, because the bug is two live object_ids on one
    person in the same moment. Picks the highest-IoU committed track."""
    hold =UID_HOLD_SECONDS if hold is None else hold
    cut =now -hold
    best_pid ,best_iou =None ,iou_thr
    for k ,p in tracker_to_person .items ():
        if p .startswith ("T"):
            continue
        if k [0 ]!=src_idx :
            continue
        if exclude_tkey is not None and k ==exclude_tkey :
            continue
        if tracker_last_seen .get (k ,0.0 )<cut :
            continue
        if p not in DB ._meta_cache :
            continue
        # latest known box for this committed UID on this camera
        for (rp ,rb ,_rc ,_rts )in recent_local_tracks .get (src_idx ,[]):
            if rp !=p :
                continue
            _i =_iou (box ,rb )
            if _i >=best_iou :
                best_iou =_i
                best_pid =p
            break
    return best_pid


def _register_local_track (src_idx ,pid ,box ,now ):
    """Remember a committed UID's latest position so a re-minted tracker id at
    the same spot can inherit it (anti-flicker)."""
    cx ,cy =_box_center (box )
    lst =recent_local_tracks [src_idx ]
    for i ,(p ,_b ,_c ,_ts )in enumerate (lst ):
        if p ==pid :
            lst [i ]=(pid ,box ,(cx ,cy ),now )
            return
    lst .append ((pid ,box ,(cx ,cy ),now ))


def _repoint_local_tracks (drop_pid ,keep_pid ):
    """After DB.merge_persons(keep, drop), drop_pid NO LONGER EXISTS. Every
    structure that can hand that id back out must be repointed, or a later
    _recover_local_uid() will resurrect the dead UID and the OSD will show two
    different names for one identity (the cam-02 FU7OMP / cam-04 5QU7FE bug).
    Rewrites entries in place and folds duplicates (keep the most recent)."""
    for _si in list (recent_local_tracks .keys ()):
        lst =recent_local_tracks [_si ]
        if not any (p ==drop_pid for (p ,_b ,_c ,_ts )in lst ):
            continue
        merged ={}
        for (p ,b ,c ,ts )in lst :
            p =keep_pid if p ==drop_pid else p
            prev =merged .get (p )
            if prev is None or ts >=prev [3 ]:
                merged [p ]=(p ,b ,c ,ts )
        recent_local_tracks [_si ]=list (merged .values ())
    # Provisional buffers / per-track bookkeeping keyed by the dead id.
    for _d in (_last_gallery_update ,):
        for _k in [k for k in list (_d .keys ())
        if isinstance (k ,tuple )and drop_pid in k ]:
            _d .pop (_k ,None )
    # Carry the dropped identity's demographic evidence over to the survivor.
    _dv =uid_demo_votes .pop (drop_pid ,None )
    if _dv :
        for _lbl ,_n in _dv .items ():
            uid_demo_votes [keep_pid ][_lbl ]+=_n
    # Fold the dropped identity's CSV dwell sessions into the survivor.
    _repoint_sessions (drop_pid ,keep_pid )


def _recover_local_uid (src_idx ,box ,now ,active_pids ):
    """If a fresh tracker id's box matches a recently-lost UID on the same
    camera, return that UID to inherit. active_pids are UIDs currently held by
    OTHER live tracks (never steal an in-use identity)."""
    cx ,cy =_box_center (box )
    diag =math .hypot (FRAME_W ,FRAME_H )
    best_pid ,best_score =None ,0.0
    keep =[]
    for (p ,b ,c ,ts )in recent_local_tracks [src_idx ]:
        if now -ts >RECOVER_MAX_GAP :
            continue                      # too old, drop it
        # A UID that no longer exists in the DB was merged away; it must never
        # be handed back out or we re-split an identity we just unified.
        if p not in DB ._meta_cache :
            continue
        keep .append ((p ,b ,c ,ts ))
        if p in active_pids :
            continue                      # in use by a live track; don't steal
        iou =_iou (box ,b )
        center_d =math .hypot (cx -c [0 ],cy -c [1 ])/diag
        # Score IoU and center-proximity on one scale. The old code ranked by
        # IoU alone, so a center-only match (iou==0) could satisfy the
        # condition yet never beat best_iou=0.0 and was silently discarded.
        prox =max (0.0 ,1.0 -(center_d /RECOVER_MAX_CENTER_FRAC ))
        if iou >=RECOVER_IOU or center_d <=RECOVER_MAX_CENTER_FRAC :
            score =max (iou ,prox *RECOVER_IOU )
            if score >best_score :
                best_score =score
                best_pid =p
    recent_local_tracks [src_idx ]=keep
    return best_pid


class ReIDProbe (BatchMetadataOperator ):
    def handle_metadata (self ,batch_meta ):
        try :
            _handle_batch (batch_meta )
        except Exception as exc :
            print (f"[PROBE ERROR] {exc }")


def _mcmtt_allows (pid ,cam ,now ):
    """Spatiotemporal gate: reject a candidate identity that is physically
    impossible for this camera right now (still live on another camera, or not
    enough time to have walked here). Returns (ok, reason)."""
    if GALLERY is None :
        return True ,None
    idn =GALLERY .get (pid )
    if idn is None :
        return True ,None
    return GALLERY ._eligible (idn ,cam ,now )


def _uids_live_on_other_cameras (src_idx ,now ,hold =None ):
    """Ground-truth occupancy: the set of committed UIDs currently rendered by a
    LIVE, PRESENT track on a DIFFERENT camera than src_idx.

    THIS is the hard defence against one global UID being painted on several
    different people at once (the SFK6AT-on-three-men bug): a UID visibly held by
    a live track on another camera is INELIGIBLE to be assigned to a detection
    here, unless enough time has passed for the person to have physically walked
    over (handled separately by the transit gate). Overlapping-FOV cameras (same
    physical space, if any are configured) are EXEMPT, because the same person
    genuinely does appear on both at once there.

    Occupancy is measured from tracker_present_ts — the last time the other
    camera's track had a REAL, confident detection — NOT tracker_last_seen, which
    keeps advancing while NvDCF coasts the box in shadow for ~17s after the
    person has already left that camera. Using that shadow-inflated timestamp
    with the 18s UID_HOLD window was exactly what blocked the cam-07 -> cam-08
    hand-off and minted a second UID. The window here is the short
    CROSS_CAM_BUSY_WINDOW, so only genuine same-instant presence blocks a match.
    """
    hold =CROSS_CAM_BUSY_WINDOW if hold is None else hold
    cut =now -hold
    cam_here =camera_map .get (src_idx )
    busy =set ()
    for k ,p in tracker_to_person .items ():
        if p .startswith ("T"):
            continue
        if k [0 ]==src_idx :
            continue
        # Real-presence timestamp; fall back to last_seen only if presence was
        # never recorded (e.g. a build with no per-object tracker confidence).
        present =tracker_present_ts .get (k )
        if present is None :
            present =tracker_last_seen .get (k ,0.0 )
        if present <cut :
            continue
        cam_there =camera_map .get (k [0 ])
        # Same person can legitimately be on both if the two cameras overlap.
        if GALLERY is not None and cam_here and cam_there \
        and GALLERY ._cameras_overlap (cam_here ,cam_there ):
            continue
        busy .add (p )
    return busy


def _mcmtt_observe (pid ,vec ,cam ,now ):
    """Record an accepted observation into the global gallery (multi-view)."""
    if GALLERY is None or vec is None :
        return
    try :
        GALLERY .update (pid ,vec ,cam ,now )
    except Exception :
        pass


def _try_cross_camera_merge (tkey ,pid ,cam ,src_idx ,now ):
    """Tier-2: once a committed identity has enough ReID vectors, look for the
    SAME person already committed on ANOTHER camera and merge the two UIDs.
    No-op if ReID vectors are absent. Returns the (possibly new) pid."""
    buf =_track_emb_buffer .get (tkey )
    if not buf or len (buf .get ("vecs",[]))<REID_MERGE_MIN_VECS :
        return pid
    if now -_last_gallery_update .get (("merge",tkey ),0.0 )<REID_MERGE_INTERVAL :
        return pid
    _last_gallery_update [("merge",tkey )]=now

    avg =_l2_mean_checked (buf ["vecs"])
    if avg is None :
        return pid

    # FURNITURE GUARD (fixed). The old rule refused a cross-camera merge for ANY
    # track that had not moved (`not tracker_ever_moved`). Every person in this
    # deployment is a SEATED office worker who rarely accumulates the 6 net-motion
    # frames needed to latch `tracker_ever_moved`, so this single line blocked
    # EVERY hand-off ("hand-offs succeeded: 0", "NO UID seen on >1 camera") even
    # though ReID separation was excellent. The real furniture signal is not
    # "never moved" — it is "looks like a learned fixture (empty chair / bike)".
    # A seated person's crop does NOT match a fixture prototype; a parked bike or
    # empty chair does. So: block the merge only when this track's centroid still
    # matches a learned fixture appearance. A motion-confirmed person is always
    # allowed (they are proven human); a still track that is NOT a known fixture
    # is allowed to re-join its own cross-camera identity. This is what lets the
    # blue-shirt / white-shirt seated people hand off between overlapping cameras.
    if SUPPRESS_STATIC_NEW and not tracker_ever_moved .get (tkey ,False ):
        try :
            _flook ,_fsim =DB .is_fixture_appearance (avg ,cam =cam )
        except Exception :
            _flook ,_fsim =(False ,0.0 )
        if _flook :
            return pid   # looks like a known fixture on this camera -> not a person

    # Candidates: committed identities currently or recently live on OTHER
    # cameras. Exclude our own pid and same-camera identities.
    exclude ={p for k ,p in tracker_to_person .items ()
    if k [0 ]==src_idx and not p .startswith ("T")}
    exclude .add (pid )
    # MCMTT spatiotemporal exclusion: also exclude any identity that is
    # physically impossible here right now (still live on another camera, or
    # too soon to have travelled). THIS is the rule that stops an empty chair
    # on cam-05 from taking the UID of a person currently sitting on cam-04.
    for _p in list (DB ._meta_cache .keys ()):
        if _p in exclude :
            continue
        _ok ,_why =_mcmtt_allows (_p ,cam ,now )
        if not _ok :
            exclude .add (_p )
    # GROUND-TRUTH simultaneous-occupancy exclusion. A UID currently drawn on a
    # LIVE track on another camera is NOT a hand-off candidate: the person is
    # still physically there, so matching them here would paint one UID on two
    # different people at the same instant (the SFK6AT-on-three-men bug). A real
    # hand-off is the opposite case — the other-camera track has GONE (the person
    # left), so its UID is no longer in this live set and remains matchable.
    exclude |=_uids_live_on_other_cameras (src_idx ,now )
    matched ,score =DB .match (avg ,cam =cam ,
    threshold =SIMILARITY_THRESHOLD ,
    soft_threshold =SOFT_THRESHOLD ,
    cross_camera_threshold =CROSS_CAMERA_THRESHOLD ,
    soft_window =SOFT_WINDOW ,
    exclude =exclude ,
    cross_camera_floor =CROSS_CAMERA_FLOOR ,
    cross_margin =CROSS_MARGIN )
    if matched is None or matched ==pid :
        return pid

    other_cams =DB .cams_of (matched )
    if cam in other_cams and len (other_cams )<=1 :
        # same-camera identity; not a cross-camera hand-off
        return pid

    # Merge our just-minted local UID into the older cross-camera identity.
    keep ,drop =matched ,pid
    kfirst ,_ =DB .first_seen_of (keep )
    dfirst ,_ =DB .first_seen_of (drop )
    if kfirst and dfirst and dfirst <kfirst :
        keep ,drop =drop ,keep

    # Back-off: if this exact pair failed to merge recently, don't hammer it
    # every frame (this produced hundreds of identical error lines before).
    _pair =(keep ,drop )
    if now -_merge_failures .get (_pair ,0.0 )<MERGE_RETRY_BACKOFF :
        return pid

    if DB .merge_persons (keep ,drop ):
        _merge_failures .pop (_pair ,None )
        _cam_handoffs ["ok"]+=1
        # Repoint every tracker entry that referenced the dropped id.
        for k ,p in list (tracker_to_person .items ()):
            if p ==drop :
                tracker_to_person [k ]=keep
        # CRITICAL: also repoint the spatial recovery map and the MCMTT gallery.
        # Without this, `drop` survives in recent_local_tracks and is handed back
        # out by _recover_local_uid the moment this tracker id is re-minted,
        # re-splitting the identity we just merged.
        _repoint_local_tracks (drop ,keep )
        if GALLERY is not None :
            try :
                GALLERY .merge (keep ,drop )
            except Exception :
                try :
                    GALLERY .drop (drop )
                except Exception :
                    pass
        print (f"[X-CAM] {tkey } {drop }->{keep } ({score :.3f}) on {cam } "
        f"| cross-camera hand-off (cams now {sorted (DB .cams_of (keep ))})")
        return keep
    _merge_failures [_pair ]=now
    return pid

def _reap_same_cam_duplicates (src_idx ,cam ,now ):
    """Immediately merge two DIFFERENT committed UIDs held by two live tracks on
    the SAME camera whose appearances match above the same-cam bar.

    Direct fix for the OKL6HU/M77XWP split: a re-minted track on cam4 that
    failed to re-match its own multi-cam UID (gallery pollution pushed the score
    below the cross-cam bar) mints a SECOND UID, and the two coexist for many
    seconds until the periodic cross-camera merge happens to fire. Checking
    same-camera duplicates every commit collapses it in the same frame. Only
    fires when BOTH UIDs are real (non-T), both live, and their gallery centroids
    agree above max(SIMILARITY_THRESHOLD, DUP_REAP_MIN_SIM) — two genuinely
    different people (centroids near the ~0.4 stranger mean) are never fused."""
    live ={}
    for k ,p in tracker_to_person .items ():
        if k [0 ]!=src_idx or p .startswith ("T"):
            continue
        if tracker_last_seen .get (k ,0.0 )<now -UID_HOLD_SECONDS :
            continue
        if p not in live or tracker_last_seen .get (k ,0 )>tracker_last_seen .get (live [p ],0 ):
            live [p ]=k
    pids =[p for p in live if p in DB ._emb_cache and DB ._emb_cache [p ]]
    if len (pids )<2 :
        return
    cents ={}
    for p in pids :
        c =_l2_mean_checked (DB ._emb_cache [p ])
        if c is not None :
            cents [p ]=c
    pids =[p for p in pids if p in cents ]
    best =None
    for i in range (len (pids )):
        for j in range (i +1 ,len (pids )):
            a ,b =pids [i ],pids [j ]
            s =float (np .dot (cents [a ],cents [b ]))
            if s >=max (SIMILARITY_THRESHOLD ,DUP_REAP_MIN_SIM )and (best is None or s >best [2 ]):
                best =(a ,b ,s )
    if best is None :
        return
    a ,b ,s =best
    kfirst ,_ =DB .first_seen_of (a )
    dfirst ,_ =DB .first_seen_of (b )
    keep ,drop =(a ,b )if (kfirst or 0 )<=(dfirst or 0 )else (b ,a )
    if DB .merge_persons (keep ,drop ):
        for k ,p in list (tracker_to_person .items ()):
            if p ==drop :
                tracker_to_person [k ]=keep
        _repoint_local_tracks (drop ,keep )
        if GALLERY is not None :
            try :
                GALLERY .merge (keep ,drop )
            except Exception :
                try :GALLERY .drop (drop )
                except Exception :pass
        print (f"[DEDUP] same-cam merge {drop }->{keep } ({s :.3f}) on {cam } "
        f"(collapsed a split identity)")

def _handle_batch (batch_meta ):

    global log_file ,writer ,_frame_count ,_person_count ,_reid_count
    global _raw_obj_count ,_bad_box_count ,_bad_box_logged ,_good_box_logged
    global _src_idx_checked

    maybe_rotate ()
    maybe_cleanup ()

    _person_count_local =[0 ]
    _reid_count_local =[0 ]
    _raw_obj_local =[0 ]
    _bad_box_local =[0 ]

    seen_per_cam =defaultdict (set )    # per-frame de-dup for health reports
    tot_per_cam =defaultdict (int )
    partial_cnt =defaultdict (int )

    # Load the whole-batch ReID tensor once per batch so per-object index-style
    # metadata can be resolved into feature vectors inside extract_reid().
    global _batch_reid_features ,_batch_reid_map
    # ReID is read per-object via the native pyservicemaker obj_reid_items /
    # as_obj_reid().featureVector() path (see extract_reid). No batch-tensor or
    # pyds pointer walking — those segfaulted on this build.
    _batch_reid_features =None
    _batch_reid_map =None

    global _batch_diag_done
    if not _batch_diag_done :
        _batch_diag_done =True
        try :
            _bumi =getattr (batch_meta ,"user_meta_items",None )
            _bu =[]
            if callable (_bumi ):
                # DS 9.1 requires a meta_type argument; enumerate the ReID types.
                for _mt in (NVDS_TRACKER_BATCH_REID_META ,NVDS_TRACKER_OBJ_REID_META ):
                    try :
                        _bu +=list (_bumi (_mt )or [])
                    except TypeError :
                        _bu =list (_bumi ()or [])
                        break
            print (f"[REID-DIAG] batch user-meta items (reid types) count={len (_bu )}")
            for _u in _bu [:5 ]:
                print (f"[REID-DIAG]   batch user-meta meta_type={getattr (_u ,'meta_type',None )}")
            if _batch_reid_features is not None :
                print (f"[REID-DIAG] batch ReID tensor loaded: shape="
                f"{_batch_reid_features .shape }")
            else :
                print ("[REID-DIAG] no batch ReID tensor this batch — objects must "
                "carry inline ReID vectors, or the tracker is not emitting ReID.")
        except Exception as _e :
            print (f"[REID-DIAG] batch user_meta access failed: {_e }")

    for frame_meta in batch_meta .frame_items :
        try :

            src_idx =int (frame_meta .pad_index )
            cam =camera_map .get (src_idx ,f"cam{src_idx }")
            _src_idx_seen [src_idx ]+=1
            if not _src_idx_checked :
                _sid =getattr (frame_meta ,"source_id",None )
                _bid =getattr (frame_meta ,"batch_id",None )
                print (f"[SRC-INDEX] pad_index={src_idx } source_id={_sid } "
                f"batch_id={_bid } -> using pad_index ({cam })")
                _src_idx_checked =True
            _cam_frames [cam ]+=1

            _suppressed_oids =suppress_overlaps (frame_meta ,iou_thresh =0.70 )
            # ^ was 0.5. This is an app-level NMS ON TOP of the detector's own
            # NMS. At 0.5 it suppressed one of two people standing close in the
            # alley (their boxes overlap >0.5), hiding real people in groups.
            # 0.70 only removes near-duplicate boxes on the SAME person, so
            # multiple people entering together are all kept.

            _display_labels =[]
            for obj in frame_meta .object_items :
                try :
                    _raw_obj_local [0 ]+=1
                    _cam_raw [cam ]+=1

                    if obj .object_id in _suppressed_oids :
                        _hide_obj (obj )
                        continue

                    cid =int (obj .class_id )
                    _class_counts [cid ]+=1
                    if not (0 <=cid <len (LABELS ))and cid not in _unknown_cids :
                        _unknown_cids .add (cid )
                        print (f"[CLASS-WARN] detector emitted class_id={cid } but "
                        f"labels.txt only defines 0..{len (LABELS )-1 }. Every "
                        "detection of this class is being DISCARDED.")
                    demo_label =(LABELS [cid ]if 0 <=cid <len (LABELS )else "Person")

                    _tk =(src_idx ,int (obj .object_id ))
                    tracker_seen [_tk ]+=1
                    established =(_tk in tracker_to_person or
                    tracker_seen [_tk ]>=ESTABLISH_AFTER_SEEN or
                    _tk in _track_emb_buffer )

                    if not is_person_class (cid ):
                        _class_rejected [cid ]+=1
                        _hide_obj (obj )
                        continue

                    _conf_bar =(MIN_DET_CONFIDENCE_KEEP if established
                    else MIN_DET_CONFIDENCE )
                    if float (obj .confidence )<_conf_bar :
                        _hide_obj (obj )
                        continue

                    vec =extract_reid (obj )

                    if vec is not None :
                        _ident =tracker_to_person .get (_tk )
                        if _ident is None or _ident .startswith ("T"):
                            _ident =_tk
                        if not reid_health_record (vec ,_ident ):
                            vec =None
                    else :
                        _reid_health ["none"]+=1

                    r =obj .rect_params
                    raw_w ,raw_h =int (r .width ),int (r .height )
                    raw_cx =float (r .left )+raw_w /2.0
                    raw_cy =float (r .top )+raw_h /2.0

                    _w_bar =MIN_BOX_W_KEEP if established else MIN_BOX_W
                    _h_bar =MIN_BOX_H_KEEP if established else MIN_BOX_H
                    if raw_w <_w_bar or raw_h <_h_bar :
                        continue

                    box_aspect =raw_h /max (raw_w ,1 )
                    area_frac =(raw_w *raw_h )/float (FRAME_W *FRAME_H )

                    aspect_bar =(MIN_PERSON_ASPECT_KEEP if established
                    else MIN_PERSON_ASPECT_NEW )
                    too_wide =box_aspect <aspect_bar
                    too_big =area_frac >MAX_PERSON_AREA_FRAC
                    spans_frame =(raw_w >0.80 *FRAME_W or
                    raw_h >0.97 *FRAME_H )
                    if too_wide or too_big or spans_frame :
                        _bad_box_local [0 ]+=1
                        if _bad_box_logged <8 :
                            reason =("too_wide"if too_wide else
                            "too_big"if too_big else "spans_frame")
                            print (f"[BOX-REJECT] cam={cam } "
                            f"w={raw_w } h={raw_h } aspect={box_aspect :.2f} "
                            f"area={area_frac :.2f} reason={reason }")
                            _bad_box_logged +=1
                        _hide_obj (obj )
                        continue

                    if _good_box_logged <8 :
                        print (f"[BOX-ACCEPT] cam={cam } "
                        f"w={raw_w } h={raw_h } aspect={box_aspect :.2f} "
                        f"area={area_frac :.2f}")
                        _good_box_logged +=1

                    left ,top ,w ,h =expand_box (r )
                    # "Partial" means a TRUNCATED crop (person cut off by the
                    # frame edge), not simply a short box. A SEATED person is
                    # legitimately near-square — the log shows the cam-04 desk
                    # worker at aspect 1.03 and the cam-07 man at 1.36, both
                    # below the 1.6 bar — so keying off aspect alone marked every
                    # seated person partial and starved them of ReID vectors.
                    # A box that does not touch any frame edge is a COMPLETE
                    # view of whatever is there, whatever its shape.
                    _sitting =("sitting"in LABELS [cid ].lower ()
                    if 0 <=cid <len (LABELS )else False )
                    is_partial =((h /max (w ,1 ))<PARTIAL_ASPECT_RATIO
                    and not _sitting )

                    edge_margin =4
                    touches_edge =(int (r .left )<=edge_margin or
                    int (r .top )<=edge_margin or
                    int (r .left +r .width )>=FRAME_W -edge_margin or
                    int (r .top +r .height )>=FRAME_H -edge_margin )
                    # Only a box that BOTH touches an edge AND is truncated is
                    # unusable for identity. A fully-inside box is always usable.
                    good_for_id =(vec is not None and
                    raw_w >=MIN_BOX_W and raw_h >=MIN_BOX_H and
                    not (touches_edge and is_partial ))

                    if not _NO_META_WRITE :
                        r .left =left
                        r .top =top
                        r .width =w
                        r .height =h
                        # NvDCF's visual tracker estimates a bbox rotation and
                        # writes it to rect_params.rotation_angle, which makes the
                        # OSD draw a tilted (diagonal) box. Force it upright.
                        for _rot_attr in ("rotation_angle","rotation"):
                            if hasattr (r ,_rot_attr ):
                                try :
                                    setattr (r ,_rot_attr ,0.0 )
                                except Exception :
                                    pass
                        obj .class_id =0

                    tkey =(src_idx ,int (obj .object_id ))
                    now =time .time ()
                    # Real-presence timestamp for the cross-camera occupancy guard.
                    # A confident NvDCF detection (>= PRESENCE_CONF) means the
                    # person is genuinely on THIS camera now; a low-confidence box
                    # is the tracker coasting in shadow after they left. Only the
                    # former refreshes tracker_present_ts, so a person who walked
                    # from cam-07 to cam-08 stops "occupying" cam-07 within
                    # CROSS_CAM_BUSY_WINDOW and can hand off to cam-08.
                    try :
                        _pconf =float (getattr (obj ,"tracker_confidence",1.0 ))
                    except Exception :
                        _pconf =1.0
                    if _pconf >=PRESENCE_CONF :
                        tracker_present_ts [tkey ]=now
                    # Empty-chair / fixture flicker guard: when a provisional
                    # (uncommitted) detection is judged to be furniture, we not
                    # only withhold a UID (below) but also HIDE its box, so an
                    # intermittently-detected empty chair never blinks a
                    # provisional box on/off. Set True in the static-suppression
                    # block; consumed at draw time.
                    _furniture_hide =False
                    _person_count_local [0 ]+=1
                    _cam_persons [cam ]+=1
                    if vec is not None :
                        _reid_count_local [0 ]+=1

                    tracker_hits [tkey ]+=1
                    if tracker_hits [tkey ]<MIN_TRACK_HITS :
                        _hide_obj (obj )
                        tracker_last_seen [tkey ]=now
                        continue

                    if STATIC_FLAG_ENABLE :
                        _is_static =is_static_box (tkey ,raw_cx ,raw_cy ,now )
                        tracker_is_static [tkey ]=_is_static
                        if tkey not in tracker_first_seen_ts :
                            tracker_first_seen_ts [tkey ]=now
                        # DO NOT infer motion from `not _is_static` here.
                        # is_static_box() returns False for THREE different
                        # reasons: <20 samples, <3s span, or genuinely moving.
                        # The first two mean "don't know yet", not "moved". The
                        # old code treated them as motion, so ANY track — parked
                        # motorbikes, an empty chair — was flagged ever_moved
                        # ~0.5s after birth, roughly 3s before is_static_box
                        # could even form an opinion. That silently disabled the
                        # entire furniture guard and is why bikes on cam-08 and
                        # an empty chair on cam-04 were issued person UIDs.
                        # Motion is now decided ONLY by cumulative centre
                        # displacement (STATIC_MOVE_MIN_PX), which is valid from
                        # frame 1. Evaluated HERE, for every track, so that the
                        # furniture verdict is consistent on every commit path
                        # (Tier-1, reidless, recovery) instead of only inside the
                        # Tier-1 branch.
                        _fs =_track_first_center .get (tkey )
                        if _fs is None :
                            _track_first_center [tkey ]=(raw_cx ,raw_cy )
                            _fs =(raw_cx ,raw_cy )
                        # Cumulative displacement from the track's birth position.
                        _disp =math .hypot (raw_cx -_fs [0 ],raw_cy -_fs [1 ])
                        if _disp >=STATIC_MOVE_MIN_PX :
                            tracker_ever_moved [tkey ]=True

                        # PERSON vs FIXTURE — decided by MOTION HISTORY, not by
                        # detector confidence. The images proved confidence is the
                        # WRONG signal: a well-lit parked-bike row detects with
                        # HIGHER, steadier confidence than a dim seated person, so
                        # a confidence gate committed the bikes and starved the
                        # people. The one signal that truly separates them is that
                        # PEOPLE MOVE and FIXTURES DO NOT.
                        #
                        # We accumulate the peak displacement AND a running count
                        # of frames with meaningful frame-to-frame motion, over the
                        # track's whole life. A person — even a seated one — racks
                        # up small motions (leaning, reaching, shifting, arriving,
                        # eventually standing). A parked-bike row or empty chair
                        # stays pinned to the same pixels indefinitely.
                        _pk =_track_peak_disp .get (tkey ,0.0 )
                        if _disp >_pk :
                            _track_peak_disp [tkey ]=_disp
                        # Window net-motion: compare the mean of the OLDER half of
                        # the window to the mean of the NEWER half. Jitter cancels
                        # in each mean; sustained motion shows up as a real shift
                        # between the two half-means. This is what distinguishes a
                        # leaning/shifting person from a jittering-but-pinned box.
                        _win =_track_center_win [tkey ]
                        _win .append ((raw_cx ,raw_cy ))
                        if len (_win )>=8 :
                            _half =len (_win )//2
                            _old =list (_win )[:_half ];_new =list (_win )[_half :]
                            _oldc =(sum (p [0 ]for p in _old )/len (_old ),
                            sum (p [1 ]for p in _old )/len (_old ))
                            _newc =(sum (p [0 ]for p in _new )/len (_new ),
                            sum (p [1 ]for p in _new )/len (_new ))
                            # Count CONSECUTIVE frames of window net-motion. Real
                            # motion is sustained across a run of frames; detector
                            # jitter on a fixture crosses the threshold only
                            # sporadically, so its run length stays short. Using
                            # the max consecutive run (not a raw total) is what
                            # stops a 5px-jitter chair from accumulating enough
                            # isolated crossings to look like a person.
                            if math .hypot (_newc [0 ]-_oldc [0 ],
                            _newc [1 ]-_oldc [1 ])>=MOTION_STEP_MIN_PX :
                                _track_motion_run [tkey ]=_track_motion_run .get (tkey ,0 )+1
                                if _track_motion_run [tkey ]>tracker_motion_frames .get (tkey ,0 ):
                                    tracker_motion_frames [tkey ]=_track_motion_run [tkey ]
                            else :
                                _track_motion_run [tkey ]=0
                        _track_last_center [tkey ]=(raw_cx ,raw_cy )

                        # A track is a PERSON once it shows a sustained motion run
                        # (arrival, shifting, standing), and stays one forever
                        # (latch). This immunises anyone who moves.
                        if tracker_motion_frames .get (tkey ,0 )>=MOTION_FRAMES_FOR_PERSON :
                            tracker_ever_moved [tkey ]=True

                        # AUTO-LEARN FIXTURE ZONES. A track that has NOT been
                        # confirmed a person, has essentially never moved
                        # (peak displacement below the person threshold), and has
                        # persisted a long time is a permanent fixture (parked
                        # bikes, empty chair). Register its location so that this
                        # and future pinned detections there are suppressed. A real
                        # person is confirmed by motion before this fires, so they
                        # never register a zone; and a person walking through moves
                        # out of the cell, so an existing zone never suppresses
                        # them.
                        _age_now =now -tracker_first_seen_ts .get (tkey ,now )
                        if (not tracker_ever_moved .get (tkey ,False )
                        and _age_now >=FIXTURE_LEARN_SECONDS
                        and _track_peak_disp .get (tkey ,0.0 )<STATIC_MOVE_MIN_PX ):
                            _register_fixture (src_idx ,raw_cx ,raw_cy ,now )
                            # Also learn this fixture's APPEARANCE so it can be
                            # rejected by ReID even if it later drifts out of its
                            # zone (a chair nudged across a grid boundary). Motion
                            # alone cannot separate a still chair from a still
                            # person; appearance can. Only learn from a clean,
                            # fully-in-frame crop.
                            if vec is not None and good_for_id :
                                DB .learn_fixture (vec ,cam =cam )

                        _dscore ,_dready =_deformation_score (tkey ,raw_w ,raw_h )
                        _first_ts =tracker_first_seen_ts .get (tkey ,now )
                        if tracker_ever_moved .get (tkey ,False ):
                            tracker_deform [tkey ]=(max (_dscore ,DEFORM_PERSON_MIN ),True )
                        else :
                            tracker_deform [tkey ]=(_dscore ,_dready )

                    # NOTE: furniture (empty-chair) suppression is enforced at the
                    # UID-commit gate below (a static-since-birth track is not given
                    # a UID until it moves or clears the furniture window), NOT by a
                    # hard skip here — so a genuinely still, newly-seated person is
                    # still drawn provisionally while we decide.

                    if tkey in tracker_to_person :
                        pid =tracker_to_person [tkey ]
                        if pid .startswith ("T"):
                            # ---- Tier 1: provisional -> committed global UID ----
                            buf =_track_emb_buffer .setdefault (tkey ,{"vecs":[],"t0":now })
                            if good_for_id and vec is not None :
                                buf ["vecs"].append (vec )

                            have_vecs =len (buf ["vecs"])>=1
                            hits =tracker_hits [tkey ]
                            age =now -tracker_first_seen_ts .get (tkey ,now )
                            ever_moved =tracker_ever_moved .get (tkey ,False )

                            # A track that has NEVER moved since it first appeared
                            # might be furniture (empty chair) OR a person who was
                            # already seated when the stream started. We do NOT
                            # commit such a track until it has either (a) shown
                            # real motion, or (b) persisted past the furniture
                            # window AND kept decent tracker confidence (NvDCF).
                            # A track that HAS moved is a confirmed person -> commit
                            # fast. This ordering is what actually keeps the empty
                            # chair from getting a UID.
                            _tconf =1.0
                            try :
                                _tconf =float (getattr (obj ,"tracker_confidence",1.0 ))
                            except Exception :
                                _tconf =1.0

                            # Motion was already decided above (hoisted so that
                            # every commit path shares one verdict). Just read it.
                            ever_moved =tracker_ever_moved .get (tkey ,False )

                            static_ok =True
                            if SUPPRESS_STATIC_NEW and not ever_moved :
                                # This track has not been confirmed a person by
                                # motion. Decide using the auto-learned fixture map
                                # plus a couple of escape routes so a genuinely
                                # still person is never blocked forever:
                                #
                                #   (a) motion arrives -> ever_moved latches above,
                                #       commits immediately (handles walk-in-sit,
                                #       standing, any shifting).
                                #   (b) FIXTURE ZONE: the detection sits in a
                                #       location already learned as a permanent
                                #       fixture (bikes/chair) -> suppress. This is
                                #       the robust bikes/chair killer: it is based
                                #       on WHERE the box is, learned from long
                                #       persistence-in-place, not on a fragile
                                #       per-track box-shape guess.
                                #   (c) grace: a track with SOME motion, past
                                #       STATIC_GRACE_SECONDS, commits — a person
                                #       who shifts a little but hasn't hit the full
                                #       motion count yet.
                                #   (d) a long-lived, never-moved track NOT in a
                                #       known zone still eventually registers one
                                #       (above) and is suppressed here.
                                _mframes =tracker_motion_frames .get (tkey ,0 )
                                _peak =_track_peak_disp .get (tkey ,0.0 )
                                _in_zone =_in_fixture_zone (src_idx ,raw_cx ,raw_cy ,now )
                                # APPEARANCE VETO: if this still track looks like a
                                # learned fixture (empty chair / bike) on THIS
                                # camera, refuse it regardless of position. This is
                                # what catches an office chair that jitters out of
                                # its grid cell — the pixels move but the
                                # appearance is still "chair".
                                _fix_look ,_fix_sim =(False ,0.0 )
                                if len (buf ["vecs"])>=1 :
                                    _fix_look ,_fix_sim =DB .is_fixture_appearance (
                                    buf ["vecs"][-1 ],cam =cam )
                                if _in_zone or _fix_look :
                                    static_ok =False    # known fixture (place or look)
                                    _furniture_hide =True   # don't draw an empty-chair box
                                elif (age >=STATIC_GRACE_SECONDS
                                and _mframes >=MOTION_FRAMES_GRACE ):
                                    static_ok =True     # SUSTAINED motion = person
                                elif (len (buf ["vecs"])>=EMB_DECIDE_COUNT
                                and age >=FIXTURE_LEARN_SECONDS
                                and not _fix_look and _fix_sim <SOFT_THRESHOLD
                                and (_mframes >FIXTURE_MAX_MOTION_FRAMES
                                or _peak >=STATIC_STILL_PERSON_MIN_PX )):
                                    # IMAGE-1 FIX (still, already-seated person who
                                    # never walks far — e.g. the man on the step on
                                    # cam-07). They accumulate a stable, PERSON-like
                                    # appearance that does NOT match any learned
                                    # fixture. Motion can't fully confirm them, but a
                                    # clean non-fixture embedding is positive
                                    # evidence of personhood, so let them commit
                                    # rather than waiting the full
                                    # FIXTURE_CONFIRM_SECONDS.
                                    #
                                    # COLD-START EMPTY-CHAIR FIX (CD82IZ): the old
                                    # branch required ONLY "clean, not-yet-a-known-
                                    # fixture appearance". On a FRESH gallery (DB
                                    # loaded 0 persons) NOTHING has been learned as a
                                    # fixture yet, so an empty office chair — which
                                    # the detector happily calls "Adult - Male
                                    # sitting" — trivially passed (_fix_look False,
                                    # _fix_sim low) and committed a UID on furniture
                                    # before any fixture zone existed to catch it.
                                    # The separating fact is MICRO-MOTION: a real
                                    # seated person shifts, types, turns — they
                                    # accumulate more than FIXTURE_MAX_MOTION_FRAMES
                                    # net-motion frames (or some real peak travel)
                                    # over FIXTURE_LEARN_SECONDS. A truly empty chair
                                    # produces ~0 motion frames and ~0 displacement.
                                    # Requiring that micro-motion here denies the
                                    # empty chair this escape while still admitting
                                    # the genuinely-still person, who is never
                                    # perfectly motionless for 6+ seconds. A chair
                                    # that DOES pass will also register its fixture
                                    # zone within FIXTURE_LEARN_SECONDS and be caught
                                    # on the next appearance regardless.
                                    static_ok =True
                                elif age >=FIXTURE_CONFIRM_SECONDS :
                                    # long-lived + never moved + not (yet) zoned:
                                    # treat as fixture until it either moves or its
                                    # zone registers.
                                    is_fixture =(_mframes <FIXTURE_MAX_MOTION_FRAMES
                                    and _peak <STATIC_MOVE_MIN_PX )
                                    static_ok =not is_fixture
                                    if is_fixture :
                                        _furniture_hide =True   # confirmed furniture
                                else :
                                    static_ok =False    # still deciding

                            commit_by_reid =have_vecs and static_ok and (
                            len (buf ["vecs"])>=EMB_DECIDE_COUNT
                            or now -buf ["t0"]>EMB_DECIDE_TIMEOUT )
                            # A reidless (no-appearance) commit is ONLY safe for a
                            # track we have already confirmed is a PERSON by motion.
                            # The old code allowed it for ANY static_ok track, which
                            # meant a still empty chair that happened not to be in a
                            # registered fixture zone yet (e.g. in its first ~30s,
                            # before FIXTURE_CONFIRM_SECONDS) got a UID with zero
                            # appearance evidence. That was the empty-chair commit
                            # path in image 3. Now: no motion confirmation AND no
                            # ReID vector => no UID. A genuinely still, already-
                            # seated PERSON still commits via commit_by_reid as soon
                            # as a vector arrives (which it does every frame), so
                            # this does not starve real people — it only denies the
                            # no-evidence fallback to unmoving furniture.
                            ever_moved_now =tracker_ever_moved .get (tkey ,False )
                            commit_reidless =(ALLOW_REIDLESS_COMMIT
                            and static_ok and ever_moved_now
                            and hits >=COMMIT_AFTER_HITS
                            # Only fall back to a no-appearance UID once we have
                            # genuinely waited for a vector that never came.
                            and age >=REIDLESS_COMMIT_DELAY )

                            # Final appearance veto on the COMMIT centroid: if a
                            # never-moved track's averaged embedding matches a
                            # learned fixture, it is furniture — do not commit.
                            # (Motion-confirmed people are exempt; they are known
                            # persons regardless of any coincidental similarity.)
                            if commit_by_reid and not ever_moved_now :
                                _avg_peek =_l2_mean_checked (buf ["vecs"])
                                if _avg_peek is not None :
                                    _flook ,_fsim =DB .is_fixture_appearance (
                                    _avg_peek ,cam =cam )
                                    if _flook :
                                        commit_by_reid =False
                                        _furniture_hide =True
                                        DB .learn_fixture (_avg_peek ,cam =cam )

                            if commit_by_reid :
                                avg =_l2_mean_checked (buf ["vecs"])
                                if avg is None :
                                    # bad centroid — fall back to reidless if allowed
                                    if commit_reidless :
                                        pid =DB .create_person_reidless (cam =cam )
                                        tracker_to_person [tkey ]=pid
                                        _ ,_first =DB .first_seen_of (pid )
                                        print (f"[NEW*]  {tkey } -> {pid } (no usable "
                                        f"vec) on {cam } | UID {_first }")
                                    else :
                                        _track_emb_buffer .pop (tkey ,None )
                                    tracker_last_seen [tkey ]=now
                                else :
                                    # 6s, not 2s: NvDCF keeps a track alive for
                                    # maxShadowTrackingAge (150 frames ~= 5.7s at
                                    # 26fps), so a person briefly occluded or
                                    # missed by the detector is still very much
                                    # "present". A 2s window let their UID become
                                    # claimable during that gap.
                                    active_cut =now -UID_HOLD_SECONDS
                                    # UIDs currently held by another LIVE track on
                                    # this camera must not be matchable, or two
                                    # people end up sharing one identity.
                                    #
                                    # The old condition also required
                                    # `not tracker_is_static[k]`, which released
                                    # the UID of anyone sitting still. That is
                                    # exactly how the seated man's UID (LS9XUY)
                                    # was handed to the man who walked up and
                                    # stood next to him, leaving the seated man to
                                    # re-commit under a brand-new UID.
                                    #
                                    # Being stationary is NOT evidence of being
                                    # furniture — tracker_ever_moved is the real
                                    # person/furniture verdict, so only a track
                                    # that was never confirmed as a person may
                                    # have its UID released.
                                    other_pids ={
                                    p for k ,p in tracker_to_person .items ()
                                    if k !=tkey and not p .startswith ("T")
                                    and k [0 ]==src_idx
                                    and tracker_last_seen .get (k ,0 )>=active_cut
                                    and tracker_ever_moved .get (k ,False )}
                                    # MCMTT spatiotemporal exclusion (see above).
                                    for _p in list (DB ._meta_cache .keys ()):
                                        if _p in other_pids :
                                            continue
                                        _ok ,_why =_mcmtt_allows (_p ,cam ,now )
                                        if not _ok :
                                            other_pids .add (_p )
                                    # GROUND-TRUTH occupancy: any UID currently
                                    # drawn on a live track on ANOTHER camera is
                                    # off-limits here. This is the hard stop that
                                    # prevents one global UID (e.g. SFK6AT) being
                                    # assigned to a DIFFERENT person while it is
                                    # visibly in use elsewhere.
                                    other_pids |=_uids_live_on_other_cameras (src_idx ,now )
                                    matched_pid ,score =DB .match (
                                    avg ,cam =cam ,
                                    threshold =SIMILARITY_THRESHOLD ,
                                    soft_threshold =SOFT_THRESHOLD ,
                                    cross_camera_threshold =CROSS_CAMERA_THRESHOLD ,
                                    soft_window =SOFT_WINDOW ,
                                    exclude =other_pids ,
                                    posture_threshold =POSTURE_THRESHOLD ,
                                    posture_window =POSTURE_WINDOW_SEC ,
                                    cross_camera_floor =CROSS_CAMERA_FLOOR ,
                                    cross_margin =CROSS_MARGIN )
                                    if matched_pid is not None :
                                        seen =DB .cams_of (matched_pid )
                                        xcam =bool (seen )and cam not in seen
                                        pid =matched_pid
                                        DB .touch (pid ,cam =cam )
                                        _mcmtt_observe (pid ,avg ,cam ,now )
                                        if DB .add_embedding (pid ,avg ,cam =cam ):
                                            _reid_health ["db_writes"]+=1
                                        else :
                                            _reid_health ["db_rejects"]+=1
                                        if xcam :
                                            _cam_handoffs ["ok"]+=1
                                        _ ,_first =DB .first_seen_of (pid )
                                        print (f"[{'X-CAM'if xcam else 'RE-ID'}] "
                                        f"{tkey } -> {pid } ({score :.3f}) on {cam } "
                                        f"| UID first seen {_first }")
                                    else :
                                        # No gallery match: try same-camera spatial
                                        # recovery before minting a brand-new UID.
                                        _cur_box =(float (r .left ),float (r .top ),
                                        float (r .width ),float (r .height ))
                                        # Same UID_HOLD_SECONDS reservation as
                                        # above: a UID in use by another live
                                        # track must never be recovered onto this
                                        # one. The old 1.0s window was far shorter
                                        # than the tracker's own shadow age.
                                        _active ={p for k ,p in tracker_to_person .items ()
                                        if k !=tkey and not p .startswith ("T")
                                        and k [0 ]==src_idx
                                        and tracker_last_seen .get (k ,0 )>=now -UID_HOLD_SECONDS }
                                        _rec =_recover_local_uid (src_idx ,_cur_box ,now ,_active )
                                        if _rec is not None :
                                            pid =_rec
                                            DB .touch (pid ,cam =cam )
                                            DB .add_embedding (pid ,avg ,cam =cam )
                                            _ ,_first =DB .first_seen_of (pid )
                                            print (f"[RECOVER] {tkey } -> {pid } on {cam } "
                                            f"(inherited lost ID + attached embedding) "
                                            f"| UID {_first }")
                                        else :
                                            pid =DB .create_person (avg ,cam =cam )
                                            _ ,_first =DB .first_seen_of (pid )
                                            print (f"[NEW]   {tkey } -> {pid } "
                                            f"(best {score :.3f} < thr) on {cam } "
                                            f"| UID {_first }")
                                    tracker_to_person [tkey ]=pid
                                    _track_emb_buffer .pop (tkey ,None )
                            elif commit_reidless :
                                # No ReID vector yet but the track is solid.
                                # First try to INHERIT a recently-lost UID at this
                                # spot (anti-flicker); else mint a new local UID.
                                _cur_box =(float (r .left ),float (r .top ),
                                float (r .width ),float (r .height ))
                                _active ={p for k ,p in tracker_to_person .items ()
                                if k !=tkey and not p .startswith ("T")
                                and k [0 ]==src_idx
                                and tracker_last_seen .get (k ,0 )>=now -UID_HOLD_SECONDS }
                                _rec =_recover_local_uid (src_idx ,_cur_box ,now ,_active )
                                if _rec is not None :
                                    pid =_rec
                                    DB .touch (pid ,cam =cam )
                                    tracker_to_person [tkey ]=pid
                                    _ ,_first =DB .first_seen_of (pid )
                                    print (f"[RECOVER] {tkey } -> {pid } on {cam } "
                                    f"(inherited recently-lost local ID; no new UID) "
                                    f"| UID {_first }")
                                else :
                                    # DUP-COMMIT GUARD. Before minting a brand-new
                                    # reidless UID, check whether a committed UID is
                                    # ALREADY live at (nearly) this box on this
                                    # camera. The tracker can briefly hold two
                                    # object_ids on one person (a re-mint racing the
                                    # shadow track); without this guard the second
                                    # object_id mints a SECOND reidless UID next to
                                    # the first (the TTE69K/HG138V split seen seconds
                                    # apart on cam1-ch701). If we find an overlapping
                                    # committed track, inherit it instead of minting.
                                    _dup =_committed_uid_overlapping_box (
                                    src_idx ,_cur_box ,now ,exclude_tkey =tkey )
                                    if _dup is not None :
                                        pid =_dup
                                        DB .touch (pid ,cam =cam )
                                        tracker_to_person [tkey ]=pid
                                        _ ,_first =DB .first_seen_of (pid )
                                        print (f"[RECOVER] {tkey } -> {pid } on {cam } "
                                        f"(inherited overlapping committed UID; no "
                                        f"duplicate minted) | UID {_first }")
                                    else :
                                        pid =DB .create_person_reidless (cam =cam )
                                        tracker_to_person [tkey ]=pid
                                        _ ,_first =DB .first_seen_of (pid )
                                        print (f"[NEW-LOCAL] {tkey } -> {pid } on {cam } "
                                        f"(committed without ReID; will merge cross-cam "
                                        f"when vectors arrive) | UID {_first }")
                                # keep buffer so vectors keep accumulating for merge
                        else :
                            # ---- committed identity: enrich + cross-camera merge ----
                            DB .touch (pid ,cam =cam )
                            # Keep the MCMTT gallery's occupancy state current on
                            # EVERY frame this identity is visible. This is what
                            # makes "still live on another camera" accurate, and
                            # it also grows the multi-view prototypes so the same
                            # person still matches after standing up / turning.
                            _mcmtt_observe (pid ,vec ,cam ,now )
                            if vec is not None and good_for_id :
                                buf =_track_emb_buffer .setdefault (tkey ,{"vecs":[],"t0":now })
                                buf ["vecs"].append (vec )
                                if len (buf ["vecs"])>MAX_EMBEDDINGS :
                                    del buf ["vecs"][0 ]
                                _gsz =DB .gallery_size (pid )
                                _interval =(GALLERY_SEED_INTERVAL
                                if _gsz <GALLERY_SEED_TARGET
                                else GALLERY_UPDATE_INTERVAL )
                                if now -_last_gallery_update .get (tkey ,0.0 )>=_interval :
                                    _last_gallery_update [tkey ]=now
                                    if DB .add_embedding (pid ,vec ,cam =cam ):
                                        _reid_health ["db_writes"]+=1
                                    else :
                                        _reid_health ["db_rejects"]+=1
                                # Tier 2: attempt cross-camera hand-off/merge.
                                pid =_try_cross_camera_merge (tkey ,pid ,cam ,src_idx ,now )
                                tracker_to_person [tkey ]=pid
                                # Tier 3: collapse any same-camera split of this
                                # identity in the SAME frame it appears, instead
                                # of waiting for the periodic cross-cam merge.
                                _reap_same_cam_duplicates (src_idx ,cam ,now )
                                pid =tracker_to_person .get (tkey ,pid )
                    else :
                        buf ={"vecs":[vec ]if (good_for_id and vec is not None )else [],
                        "t0":now }
                        _track_emb_buffer [tkey ]=buf
                        pid =f"T{src_idx }_{int (obj .object_id )}"
                        tracker_to_person [tkey ]=pid

                    tracker_last_seen [tkey ]=now

                    # Keep the spatial recovery map fresh: remember where each
                    # committed UID currently is, so a re-minted tracker id at the
                    # same place can inherit it instead of spawning a new UID.
                    if not pid .startswith ("T"):
                        _register_local_track (src_idx ,pid ,
                        (float (r .left ),float (r .top ),float (r .width ),float (r .height )),
                        now )

                    _committed =not pid .startswith ("T")
                    if 0 <=cid <len (LABELS ):
                        tracker_class_votes [tkey ][cid ]+=1
                        tracker_posture_recent [tkey ].append (cid )
                        if _committed :
                            _d ,_ =split_label (LABELS [cid ])
                            if _d :
                                uid_demo_votes [pid ][_d ]+=1

                    votes =tracker_class_votes .get (tkey )
                    if votes :
                        best_cid =max (votes .items (),key =lambda kv :kv [1 ])[0 ]
                        base_label =(LABELS [best_cid ]
                        if 0 <=best_cid <len (LABELS )else demo_label )
                    else :
                        base_label =demo_label
                    demo_part ,base_posture =split_label (base_label )

                    # One identity = one demographic, on every camera. Posture is
                    # still decided locally (a person really can be sitting on one
                    # view and standing in another), but Male/Female/Kid comes
                    # from the pooled per-UID tally, resolved on two INDEPENDENT
                    # axes (age, gender) with hysteresis. This stops a burst of
                    # wrong "Kid - Girl" frames on a clearly adult male (image 1)
                    # from flipping the identity's shown demographic: the male
                    # evidence pooled across Adult-Male/Kid-Boy still wins the
                    # gender axis, and the rarer "Kid" age call needs a clear,
                    # well-supported majority before it is shown.
                    if _committed :
                        demo_part =resolve_uid_demo (pid ,demo_part )

                    recent =tracker_posture_recent .get (tkey )
                    posture_part =base_posture
                    if recent :
                        pcount =defaultdict (int )
                        for rc in recent :
                            if 0 <=rc <len (LABELS ):
                                _ ,p =split_label (LABELS [rc ])
                                if p :
                                    pcount [p ]+=1
                        if pcount :
                            posture_part =max (pcount .items (),
                            key =lambda kv :kv [1 ])[0 ]
                    stable_label =compose_label (demo_part ,posture_part )

                    # Empty-chair / fixture flicker fix: a provisional detection
                    # that has been judged furniture (in a learned fixture zone,
                    # or long-lived and never-moved) is HIDDEN rather than drawn.
                    # This is what stops the box blinking on an empty chair as the
                    # detector fires on it intermittently. A committed identity is
                    # never hidden, and a still-but-real person is only hidden
                    # until they move or the 30s fixture-confirm elapses, after
                    # which motion (even a small shift) reveals them immediately.
                    if _furniture_hide and pid .startswith ("T"):
                        _hide_obj (obj )
                        tracker_last_seen [tkey ]=now
                        continue

                    if pid .startswith ("T"):
                        tag =f"{stable_label } (ID...)"
                    else :
                        tag =f"{stable_label } {pid }"
                    if is_partial :
                        tag +="*"

                    if not HEADLESS :
                        _draw_obj_label (obj ,tag ,left ,max (0 ,top -12 ))

                    if pid not in seen_per_cam [cam ]:
                        seen_per_cam [cam ].add (pid )
                        tot_per_cam [cam ]+=1
                        if is_partial :
                            partial_cnt [cam ]+=1
                    if not pid .startswith ("T"):
                        _cam_ids [cam ].add (pid )
                        # Accumulate dwell for this committed UID on this camera.
                        # `demo_part` is the pooled one-identity-one-demographic
                        # vote, `posture_part` is the locally-decided posture
                        # (sitting/standing) used for the posture counts/times,
                        # and `now` is this object's epoch timestamp.
                        _touch_session (cam ,pid ,demo_part ,posture_part ,now )


                except StopIteration :
                    break
                except Exception as exc :
                    print (f"[OBJ ERROR] {exc }")
                    continue

        except StopIteration :
            continue
        except Exception as exc :
            print (f"[FRAME ERROR] {exc }")
            continue

    # Dwell sessions are updated per detection inside the object loop; the CSV
    # is a live snapshot of that table rewritten every FLUSH_EVERY frames (one
    # row per identity per camera, not one per frame).
    with _log_lock :
        _flush_frame_counter [0 ]+=1
        if _flush_frame_counter [0 ]>=FLUSH_EVERY :
            _flush_frame_counter [0 ]=0
            flush_sessions ()

    _frame_count +=1
    _person_count +=_person_count_local [0 ]
    _reid_count +=_reid_count_local [0 ]
    _raw_obj_count +=_raw_obj_local [0 ]
    _bad_box_count +=_bad_box_local [0 ]
    if _frame_count %DIAG_EVERY ==0 :
        global _last_diag_time
        now_t =time .time ()
        dt =now_t -_last_diag_time if _last_diag_time else 0
        fps =(DIAG_EVERY /dt )if dt >0 else 0
        _last_diag_time =now_t
        print (f"[DIAG] frames={_frame_count } fps={fps :.1f} "
        f"raw_objects={_raw_obj_count } "
        f"garbage_boxes={_bad_box_count } "
        f"valid_persons={_person_count } "
        f"reid_vectors={_reid_count } db={DB .stats ()}")
        if REID_REPORT_EVERY and _frame_count %REID_REPORT_EVERY ==0 :
            reid_health_report ()
            camera_health_report ()
            class_health_report ()
        if fps >0 and fps <3 :
            print (f"[DIAG] >>> Pipeline running SLOW ({fps :.1f} fps). The Orin "
            "Nano is overloaded. Raise detector 'interval' in "
            "config_infer.txt to 1 or 2, and/or reidExtractionInterval.")
        if _raw_obj_count ==0 :
            print ("[DIAG] >>> Parser produced ZERO objects.")
        elif _bad_box_count >0 and _person_count ==0 :
            print ("[DIAG] >>> Only garbage/full-frame boxes, no valid persons.")
        elif _person_count >0 and _reid_count ==0 :
            _np =DB .stats ()["persons"]if DB else 0
            print (f"[DIAG] >>> No ReID vectors yet, but local UIDs still commit "
            f"({_np } persons). Within-camera IDs work; cross-camera hand-off is "
            "DISABLED until ReID vectors appear. If they never appear, the tracker "
            "config (reidType/outputReidTensor) or ReID engine is the issue.")

def _hide_obj (obj ):
    if _NO_META_WRITE :
        return
    try :
        rp =obj .rect_params
        for _a in ("border_width",):
            if hasattr (rp ,_a ):
                setattr (rp ,_a ,0 )
        for _a in ("has_bg_color","set_bg_clr"):
            if hasattr (rp ,_a ):
                setattr (rp ,_a ,0 )
    except Exception :
        pass
    try :
        tp =obj .text_params
        if hasattr (tp ,"display_text"):
            tp .display_text =""
        for _a in ("set_bg_clr","set_bg_color"):
            if hasattr (tp ,_a ):
                setattr (tp ,_a ,0 )
    except Exception :
        pass

def _draw_obj_label (obj ,tag ,x ,y ):
    if _NO_META_WRITE :
        return
    try :
        rp =obj .rect_params
        if hasattr (rp ,"border_width"):
            rp .border_width =2
        for _rot_attr in ("rotation_angle","rotation"):
            if hasattr (rp ,_rot_attr ):
                try :
                    setattr (rp ,_rot_attr ,0.0 )
                except Exception :
                    pass
    except Exception :
        pass
    try :
        txt =obj .text_params
        if hasattr (txt ,"display_text"):
            txt .display_text =tag
        for _a ,_v in (("x_offset",int (x )),("y_offset",int (y ))):
            if hasattr (txt ,_a ):
                setattr (txt ,_a ,_v )
        fp =getattr (txt ,"font_params",None )
        if fp is not None :
            if hasattr (fp ,"font_name"):
                fp .font_name ="DejaVu Sans"
            if hasattr (fp ,"font_size"):
                fp .font_size =5
            fc =getattr (fp ,"font_color",None )
            if fc is not None :
                for _c ,_v in (("red",1.0 ),("green",1.0 ),("blue",0.0 ),("alpha",1.0 )):
                    if hasattr (fc ,_c ):
                        setattr (fc ,_c ,_v )
        for _a in ("set_bg_clr","set_bg_color"):
            if hasattr (txt ,_a ):
                setattr (txt ,_a ,1 )
        bg =getattr (txt ,"text_bg_clr",None )
        if bg is not None :
            for _c ,_v in (("red",0.0 ),("green",0.0 ),("blue",0.0 ),("alpha",0.75 )):
                if hasattr (bg ,_c ):
                    setattr (bg ,_c ,_v )
    except Exception :
        pass

def _tracker_lib_path ():

    env =os .environ .get ("NVDS_TRACKER_LIB")
    if env and os .path .isfile (env ):
        return env
    candidates =[
    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    "/opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so",
    ]
    for c in candidates :
        if os .path .isfile (c ):
            return c
    return candidates [0 ]

def main ():
    global log_file ,writer ,DB ,GALLERY

    print ("="*64 )
    print ("  deepstream_app_reid  —  BUILD: mcmtt-v25-empty-chair-guard (DS9.1/pyservicemaker)")
    print ("  ReID: NATIVE obj_reid_items().as_obj_reid().featureVector() — WORKING.")
    print ("  MCMTT: spatiotemporal exclusion (one identity cannot be on two")
    print ("         cameras at once) + Hungarian global assignment + multi-view")
    print ("         prototypes for viewpoint/illumination robustness.")
    print ("  v23/v24 guards RETAINED (no overlap group; ground-truth occupancy;")
    print ("       margin-gated floor; mux timeout; dup-commit guard; VPI mitig.).")
    print ("  FIX v25: cold-start EMPTY-CHAIR guard. The appearance-only still-person")
    print ("       commit escape no longer fires on a track with ~zero micro-motion,")
    print ("       so an empty office chair the detector calls 'Adult - Male sitting'")
    print ("       can NOT commit a UID just because no fixture was learned yet on a")
    print ("       fresh gallery (the CD82IZ box). A genuinely-still PERSON still")
    print ("       commits via >FIXTURE_MAX_MOTION_FRAMES micro-motion frames or")
    print ("       >=STATIC_STILL_PERSON_MIN_PX peak travel; a chair produces neither.")
    print (f"         co-occurrence {CO_OCCURRENCE_WINDOW }s | transit {MIN_TRANSIT_TIME }s "
    f"| {MAX_VIEW_PROTOS } views/identity")
    print (f"  Thresholds: same-cam {SIMILARITY_THRESHOLD } cross-cam {CROSS_CAMERA_THRESHOLD }"
    f" (soft path can no longer undercut the cross-cam bar)")
    print (f"  Coverage: detector interval=0, reidExtractionInterval=0 (restored)")
    print (f"  Classes: {'ALL '+str (len (LABELS ))+' labeled classes tracked'if TRACK_ALL_LABELED else 'person-keyword filtered'}"
    f" (labels.txt is the filter; !nonperson to exclude)")
    print (f"  Retention: {RETENTION_SECONDS /60 :.0f} min | same-cam recovery "
    f"{RECOVER_MAX_GAP :.0f}s | reidless commit delayed {REIDLESS_COMMIT_DELAY :.0f}s")
    print (f"  UID lock: a committed UID is reserved to its track for "
    f"{UID_HOLD_SECONDS :.0f}s even while the person sits perfectly still")
    print (f"  Personhood: MOTION confirms a person ({MOTION_FRAMES_FOR_PERSON } "
    f"consecutive net-motion frames >={MOTION_STEP_MIN_PX :.0f}px, or "
    f"{STATIC_MOVE_MIN_PX :.0f}px travel), latched for life")
    print (f"  Fixtures: bikes/chairs auto-learned as suppression zones after "
    f"{FIXTURE_LEARN_SECONDS :.0f}s pinned-in-place (grid {FIXTURE_GRID_PX :.0f}px, "
    f"TTL {FIXTURE_ZONE_TTL :.0f}s); still person commits by "
    f"{FIXTURE_CONFIRM_SECONDS :.0f}s fallback")
    print ("  OSD: '<class> <UID>' committed | '<class> (ID...)' still committing")
    print ("="*64 )

    DB =PersonDatabase (db_path =DB_PATH ,
    max_embeddings_per_person =MAX_EMBEDDINGS ,
    feature_size =REID_FEATURE_SIZE ,
    seed_target =GALLERY_SEED_TARGET ,
    overlap_groups =CAMERA_OVERLAP_GROUPS )
    print (f"[INFO] DB stats at startup: {DB .stats ()}")

    # --- MCMTT global gallery -------------------------------------------
    GALLERY =GlobalGallery (
    same_cam_threshold =SIMILARITY_THRESHOLD ,
    cross_cam_threshold =CROSS_CAMERA_THRESHOLD ,
    co_occurrence_window =CO_OCCURRENCE_WINDOW ,
    min_transit_time =MIN_TRANSIT_TIME ,
    max_protos =MAX_VIEW_PROTOS ,
    overlap_groups =CAMERA_OVERLAP_GROUPS )
    if CAMERA_OVERLAP_GROUPS :
        print ("[MCMTT] overlapping-FOV camera groups (co-occurrence/transit "
        "gates skipped within each):")
        for _g in CAMERA_OVERLAP_GROUPS :
            print (f"[MCMTT]   {{{', '.join (sorted (_g ))}}}")
    # Seed it from whatever identities are already persisted, so the gate works
    # from the first frame after a restart.
    _seeded =0
    for _pid ,_embs in DB ._emb_cache .items ():
        if not _embs :
            continue
        _meta =DB ._meta_cache .get (_pid ,{})
        _cam =_meta .get ("last_cam")
        GALLERY .add (_pid ,_embs [-1 ],_cam ,now =_meta .get ("last_seen",time .time ()))
        for _e in _embs [:-1 ][-MAX_VIEW_PROTOS :]:
            GALLERY .update (_pid ,_e ,_cam ,now =_meta .get ("last_seen",time .time ()))
        _seeded +=1
    print (f"[MCMTT] gallery ready: {len (GALLERY )} identities seeded ({_seeded } "
    f"with embeddings) | co-occurrence window {CO_OCCURRENCE_WINDOW }s, "
    f"min transit {MIN_TRANSIT_TIME }s, max {MAX_VIEW_PROTOS } view-prototypes")

    log_file ,writer =open_new_log ()

    reid_engine_preflight (TRACKER_CONFIG )
    class_config_preflight ("config_infer.txt")

    n_src =len (CAMERA_STREAMS )
    for i ,cam in enumerate (CAMERA_STREAMS ):
        camera_map [i ]=cam ["name"]

    tracker_lib =_tracker_lib_path ()
    print (f"[TRACKER] ll-lib-file = {tracker_lib }")

    pipeline =Pipeline ("reid-pipeline")

    _drop_fi =0
    try :
        _drop_fi =max (0 ,min (30 ,int (os .environ .get ("DROP_FRAME_INTERVAL","0"))))
    except ValueError :
        _drop_fi =0
    _reconnect =0
    try :
        _reconnect =max (0 ,int (os .environ .get ("RTSP_RECONNECT_INTERVAL","0")))
    except ValueError :
        _reconnect =0
    if _drop_fi :
        print (f"[SOURCE] drop-frame-interval={_drop_fi } (decoder drops every "
        f"{_drop_fi }th frame to cut decode load)")
    print (f"[SOURCE] rtsp-reconnect-interval={_reconnect }s "
    f"(0 disables the forced-reconnect timeout)")

    for i ,cam in enumerate (CAMERA_STREAMS ):
        _props ={
        "uri":cam ["uri"],
        "select-rtp-protocol":4 ,
        "latency":200 ,
        "udp-buffer-size":2000000 ,
        "disable-audio":1 ,
        "rtsp-reconnect-interval":_reconnect ,
        }
        if _drop_fi :
            _props ["drop-frame-interval"]=_drop_fi
        pipeline .add ("nvurisrcbin",f"src-{i }",_props )

    pipeline .add ("nvstreammux","mux",{
    "batch-size":n_src ,
    "width":FRAME_W ,
    "height":FRAME_H ,
    "live-source":1 ,
    # batched-push-timeout is how long the muxer waits to collect one buffer from
    # EVERY source before pushing a (possibly partial) batch. At 40000us (~40ms =
    # one frame at 25fps) a source that is even slightly late — a jittery RTSP
    # link like cam3-ch201 — misses the batch cycle EVERY time, so its frames are
    # muxed with no detector attention and that camera reports raw=0 while still
    # "seeing frames". Raising the timeout to ~4 frames lets the muxer wait for
    # the straggler and batch all sources together, restoring detection on the
    # laggy camera. Tune via env if a source is much slower still.
    "batched-push-timeout":int (os .environ .get ("MUX_PUSH_TIMEOUT_US","160000")),
    # Do NOT hard-sync inputs to a common clock: a slow source would then stall
    # the whole batch. 0 lets each source flow at its own cadence.
    "sync-inputs":0 ,
    "attach-sys-ts":1 ,
    })

    PGIE_ENGINE_BATCH =4
    pipeline .add ("nvinfer","pgie",{
    "config-file-path":"config_infer.txt",
    "batch-size":max (n_src ,PGIE_ENGINE_BATCH ),
    })

    pipeline .add ("nvtracker","tracker",{
    "ll-lib-file":tracker_lib ,
    "ll-config-file":TRACKER_CONFIG ,
    "tracker-width":1280 ,
    "tracker-height":704 ,
    "display-tracking-id":1 ,
    # Larger pool so per-object ReID user-meta buffers aren't dropped under load
    # (NVIDIA "Unable to acquire a user meta buffer" guidance). Default is 16.
    "user-meta-pool-size":64 ,
    })

    for i in range (n_src ):
        pipeline .link ((f"src-{i }","mux"),("","sink_%u"))
    pipeline .link ("mux","pgie","tracker")

    if HEADLESS :

        pipeline .add ("fakesink","sink",{"sync":0 ,"qos":0 ,
        "enable-last-sample":0 })
        pipeline .link ("tracker","sink")
        pipeline .attach ("tracker",Probe ("reid-probe",ReIDProbe ()))
        print ("[PIPELINE] HEADLESS mode: tracker -> fakesink (no OSD/EGL).")
    else :

        cols =min (2 ,n_src )if n_src <=2 else 2
        rows =math .ceil (n_src /cols )
        pipeline .add ("nvmultistreamtiler","tiler",
        {"rows":rows ,"columns":cols ,
        "width":1280 ,"height":720 })
        pipeline .add ("nvvideoconvert","conv",{"compute-hw":1 })
        pipeline .add ("nvdsosd","osd",
        {"process-mode":0 ,"display-text":1 })

        pipeline .add ("nv3dsink","sink",{"sync":0 ,"qos":0 })
        pipeline .link ("tracker","tiler","conv","osd","sink")

        pipeline .attach ("tracker",Probe ("reid-probe",ReIDProbe ()))
        print (f"[PIPELINE] display mode: {n_src } cams -> {rows }x{cols } tiler @1280x720")

    print ("[INFO] Pipeline built. Starting…")
    print ("[INFO] Labels marked with * are partial-body detections.")

    _stop =threading .Event ()

    def _reid_watchdog ():
        _stop .wait (30 )
        if not _reid_seen_once :
            print ("[WARN] No ReID metadata seen in first 30s. If people HAVE "
            "been in frame, check the tracker config.")

    def _rotation_timer ():
        while not _stop .wait (30 ):
            try :
                maybe_rotate ()
            except Exception as e :
                print (f"[LOG] rotation timer error: {e }")

    threading .Thread (target =_reid_watchdog ,daemon =True ).start ()
    threading .Thread (target =_rotation_timer ,daemon =True ).start ()

    exit_code =0

    def _on_message (*args ):
        try :
            message =None
            for a in args :
                if hasattr (a ,"new_state")or hasattr (a ,"old_state"):
                    message =a
                    break
            if message is None and args :
                message =args [-1 ]
            old =getattr (message ,"old_state",None )
            new =getattr (message ,"new_state",None )
            origin =getattr (message ,"origin",None )
            if old is not None and new is not None :
                if origin in ("reid-pipeline",None )or str (new )in ("PLAYING","GST_STATE_PLAYING"):
                    print (f"[STATE] {origin }: {old } -> {new }")
        except Exception :
            pass

    import signal as _signal

    _sigint_count =[0 ]

    def _sigint_handler (signum ,frame ):
        _sigint_count [0 ]+=1
        if _sigint_count [0 ]==1 :
            print ("\n[INFO] Ctrl+C received — stopping pipeline cleanly "
            "(press Ctrl+C again to force-quit)…")
            _stop .set ()
            try :
                pipeline .stop ()
            except Exception :
                pass
        else :
            print ("\n[INFO] Second Ctrl+C — forcing exit.")
            os ._exit (0 )

    _signal .signal (_signal .SIGINT ,_sigint_handler )

    try :

        try :
            _handle =pipeline .start (on_message =_on_message )
        except TypeError :
            _handle =pipeline .start ()
        _handle .wait ()

        print ("[EOS] all sources ended. Exiting non-zero so systemd restarts "
        "the process cleanly.")
        exit_code =1
    except KeyboardInterrupt :
        print ("\n[INFO] Interrupted by user — clean shutdown (no restart).")
        exit_code =0
    except Exception as e :
        print (f"[PIPELINE ERROR] {e }")
        exit_code =1
    finally :
        _stop .set ()
        try :
            pipeline .stop ()
        except Exception :
            pass
        # Give the tracker's CUDA context a moment to drain in-flight frames
        # before the process exits, to avoid the DS 9.1 Jetson shutdown segfault
        # ("GPUassert: driver shutting down ... context.cpp"). Cosmetic only.
        try :
            time .sleep (0.5 )
        except Exception :
            pass
        with _log_lock :
            try :
                flush_sessions ()          # write the final dwell snapshot
                log_file .flush ()
                log_file .close ()
            except Exception :
                pass
        print (f"[INFO] Final DB stats: {DB .stats ()}")
        reid_health_report (final =True )
        camera_health_report (final =True )
        class_health_report (final =True )
        DB .close ()

    if exit_code !=0 :
        import sys as _sys
        _sys .exit (exit_code )

if __name__ =="__main__":
    main ()
