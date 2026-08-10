import time
import sqlite3
import random
import string
import os
import numpy as np
from collections import defaultdict

# Anti-flicker: an identity seen very recently ON THE SAME CAMERA gets a small
# similarity boost so a person who briefly leaves and returns re-attaches to
# their existing UID instead of minting a new one (the NEW->MERGE churn seen in
# the logs). Applied to SAME-CAMERA candidates only — never cross-camera, where
# inflating the score would undercut the higher cross-cam bar and risk merging
# two different people.
SAME_CAM_RECENT_BOOST =float (os .environ .get ("SAME_CAM_RECENT_BOOST","0.05"))

# Contamination guard margin: when a candidate crop matches some OTHER identity's
# gallery better than its own by more than this, it is refused (it almost
# certainly belongs to that other person). Prevents one UID absorbing another
# person's crops, which was pushing the cross-identity p95 up to the cross-cam
# bar (0.83 vs 0.84) and causing wrong merges / NEW-UID churn.
CONTAM_MARGIN =float (os .environ .get ("REID_CONTAM_MARGIN","0.05"))

def _l2 (v ):
    v =np .asarray (v ,dtype =np .float32 )
    n =np .linalg .norm (v )
    return v /n if n >0 else v

def _ts_str (ts ):

    try :
        return time .strftime ("%Y-%m-%d %H:%M:%S",time .localtime (float (ts )))
    except Exception :
        return ""

class PersonDatabase :
    def __init__ (self ,db_path ="reid_persons.db",
    max_embeddings_per_person =60 ,
    feature_size =256 ,
    commit_interval =5.0 ,
    seed_target =25 ,
    overlap_groups =None ):
        self .db_path =db_path
        self .max_emb =max_embeddings_per_person
        self .feat_sz =feature_size
        self .commit_interval =commit_interval

        self .seed_target =seed_target

        # OVERLAPPING-FOV CAMERA GROUPS (list of sets of camera names). Cameras
        # in the same group view the SAME physical space, so the SAME person
        # genuinely appears on both at once. For matching that means a candidate
        # last seen on an overlapping camera is NOT a "cross-camera" candidate:
        # it must be scored against the SAME-CAMERA bar (and may take the
        # same-camera recency boost), otherwise the strict cross-cam bar blocks
        # every legitimate in-room hand-off. This is the DB-side counterpart of
        # the MCMTT gallery's overlap logic; without it, cam3<->cam4 (one room)
        # never hand off and the person collects a second UID. See match().
        self .overlap_groups =[set (g )for g in (overlap_groups or [])]

        self ._emb_cache =defaultdict (list )

        self ._mat_cache ={}

        self ._meta_cache ={}

        # --- Fixture (non-person) prototype gallery -------------------------
        # ReID centroids of things the app has decided are NOT people (empty
        # chairs, parked bikes) via the motion/persistence guard. A candidate
        # whose appearance matches one of these above FIXTURE_REJECT_THR is
        # refused a UID. This is the appearance-based half of the empty-chair
        # fix: motion alone cannot separate a still person from a still chair,
        # but the chair's *appearance* is stable and learnable. Kept in memory
        # only (rebuilt each run from the live guard) so a chair that is later
        # permanently occupied by a person ages out naturally.
        # Each entry: {"vec": np.ndarray(L2), "cam": str, "ts": float, "n": int}
        self ._fixture_protos =[]
        self ._fixture_reject_thr =float (
        __import__ ("os").environ .get ("FIXTURE_REJECT_THR","0.83"))
        self ._fixture_merge_thr =0.94   # fold near-identical protos together
        self ._fixture_ttl =float (
        __import__ ("os").environ .get ("FIXTURE_PROTO_TTL","600"))
        self ._fixture_max =256

        self ._dirty =False
        self ._last_commit =time .time ()

        self ._conn =sqlite3 .connect (self .db_path ,check_same_thread =False )
        self ._conn .execute ("PRAGMA journal_mode=WAL;")
        self ._create_tables ()
        self ._load_into_memory ()

    def _create_tables (self ):
        c =self ._conn .cursor ()

        # PERMANENT UID LEDGER. Every id ever minted is recorded here and is
        # NEVER deleted — not on prune, not on merge, not on retention rollover.
        # `persons` only holds LIVE identities, so checking uniqueness against it
        # alone allowed a UID to be reissued to a DIFFERENT person after the
        # original was pruned (30 min) or merged away. This table is the
        # authority for "has this id ever existed", which is what guarantees
        # global UID uniqueness for the lifetime of the database.
        c .execute ("""
            CREATE TABLE IF NOT EXISTS uid_ledger (
                person_id   TEXT PRIMARY KEY,
                minted_at   REAL,
                minted_str  TEXT,
                retired_at  REAL,      -- NULL while live
                retired_why TEXT       -- 'merged'|'pruned'|NULL
            )""")

        c .execute ("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id        TEXT PRIMARY KEY,  -- random 6-char identity
                created_at       REAL,              -- Unix ts, first seen
                last_seen        REAL,              -- Unix ts, most recent
                sighting_count   INTEGER,           -- frames tracked (not visits)
                first_seen_str   TEXT,              -- readable first-seen datetime
                last_seen_str    TEXT,              -- readable last-seen datetime
                last_camera      TEXT,              -- camera of most recent sighting
                num_embeddings   INTEGER            -- vectors stored for this person
            )""")
        c .execute ("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  TEXT,
                vec        BLOB,
                created_at REAL,
                created_str TEXT,                    -- readable datetime
                camera     TEXT,                     -- camera this vec came from
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )""")
        c .execute ("CREATE INDEX IF NOT EXISTS idx_emb_person ON embeddings(person_id)")

        c .execute ("""
            CREATE TABLE IF NOT EXISTS sightings (
                person_id   TEXT,
                camera      TEXT,
                first_seen  REAL,
                last_seen   REAL,
                first_str   TEXT,
                last_str    TEXT,
                count       INTEGER,
                PRIMARY KEY (person_id, camera)
            )""")
        c .execute ("CREATE INDEX IF NOT EXISTS idx_sight_person ON sightings(person_id)")
        c .execute ("CREATE INDEX IF NOT EXISTS idx_sight_camera ON sightings(camera)")

        c .execute ("""
            CREATE TABLE IF NOT EXISTS sightings (
                person_id   TEXT,
                camera      TEXT,
                first_seen  REAL,
                last_seen   REAL,
                first_str   TEXT,
                last_str    TEXT,
                count       INTEGER,
                PRIMARY KEY (person_id, camera)
            )""")
        c .execute ("CREATE INDEX IF NOT EXISTS idx_sight_person ON sightings(person_id)")
        c .execute ("CREATE INDEX IF NOT EXISTS idx_sight_camera ON sightings(camera)")
        self ._conn .commit ()
        self ._migrate_add_columns ()

    def _migrate_add_columns (self ):

        c =self ._conn .cursor ()
        existing ={row [1 ]for row in c .execute ("PRAGMA table_info(persons)")}
        for col ,decl in (("first_seen_str","TEXT"),("last_seen_str","TEXT"),
        ("last_camera","TEXT"),("num_embeddings","INTEGER")):
            if col not in existing :
                c .execute (f"ALTER TABLE persons ADD COLUMN {col } {decl }")
        eexisting ={row [1 ]for row in c .execute ("PRAGMA table_info(embeddings)")}
        for col ,decl in (("created_str","TEXT"),("camera","TEXT")):
            if col not in eexisting :
                c .execute (f"ALTER TABLE embeddings ADD COLUMN {col } {decl }")

        for pid ,ca ,ls in c .execute (
        "SELECT person_id, created_at, last_seen FROM persons").fetchall ():
            c .execute ("UPDATE persons SET first_seen_str=?, last_seen_str=? "
            "WHERE person_id=?",
            (_ts_str (ca ),_ts_str (ls ),pid ))
        self ._conn .commit ()

    def _load_into_memory (self ):
        c =self ._conn .cursor ()

        # Backfill the ledger for databases created before it existed, so
        # already-issued ids are protected from reuse on upgrade.
        c .execute (
        "INSERT OR IGNORE INTO uid_ledger(person_id, minted_at, minted_str) "
        "SELECT person_id, created_at, first_seen_str FROM persons")
        self ._uid_ever ={r [0 ]for r in c .execute (
        "SELECT person_id FROM uid_ledger")}

        for pid ,last_seen ,count ,last_cam ,created_at ,first_str in c .execute (
        "SELECT person_id, last_seen, sighting_count, last_camera, "
        "created_at, first_seen_str FROM persons"):
            self ._meta_cache [pid ]={"last_seen":last_seen ,"count":count ,
            "last_cam":last_cam ,
            "cams":set (),
            "first_seen":created_at ,
            "first_seen_str":first_str or _ts_str (created_at )}

        loaded =0
        for pid ,blob ,camera in c .execute (
        "SELECT person_id, vec, camera FROM embeddings"):
            vec =np .frombuffer (blob ,dtype =np .float32 )
            if vec .shape [0 ]==self .feat_sz :
                self ._emb_cache [pid ].append (_l2 (vec ))
                loaded +=1

                if pid in self ._meta_cache and camera :
                    self ._meta_cache [pid ]["cams"].add (camera )

        for pid in self ._emb_cache :
            if len (self ._emb_cache [pid ])>self .max_emb :
                self ._emb_cache [pid ]=self ._emb_cache [pid ][-self .max_emb :]
                self ._mat_cache .pop (pid ,None )

        print (f"[DB] Loaded {len (self ._meta_cache )} persons, "
        f"{loaded } embeddings from {self .db_path }")

    def _new_id (self )->str :
        """Mint a globally-unique id that has NEVER been used before.

        Checked against the permanent ledger (every id ever minted), not just
        the live `persons` table, so a pruned or merged-away id is never handed
        out again. The INSERT is what reserves it, so two concurrent callers
        cannot receive the same id."""
        chars =string .ascii_uppercase +string .digits
        while True :
            pid ="".join (random .choices (chars ,k =6 ))
            if pid in self ._meta_cache or pid in self ._uid_ever :
                continue
            now =time .time ()
            try :
                self ._conn .execute (
                "INSERT INTO uid_ledger(person_id, minted_at, minted_str) "
                "VALUES (?,?,?)",(pid ,now ,_ts_str (now )))
            except sqlite3 .IntegrityError :
                # Lost a race, or the id exists in a ledger row we had not
                # cached. Either way it is taken — draw again.
                self ._uid_ever .add (pid )
                continue
            self ._uid_ever .add (pid )
            return pid

    def _retire_uid (self ,pid ,why ):
        """Mark an id as no longer live. It stays in the ledger forever so it can
        never be reissued; this only records why and when it went away."""
        try :
            self ._conn .execute (
            "UPDATE uid_ledger SET retired_at=?, retired_why=? "
            "WHERE person_id=? AND retired_at IS NULL",
            (time .time (),why ,pid ))
        except Exception :
            pass

    @staticmethod
    def _cosine (a ,b ):
        a =_l2 (a );b =_l2 (b )
        return float (np .dot (a ,b ))

    def _cams_overlap (self ,cam_a ,cam_b ):
        """True if two cameras share a physical space (same overlap group), so
        the same person may legitimately appear on both at the same instant."""
        if cam_a ==cam_b :
            return True
        if not cam_a or not cam_b :
            return False
        for g in self .overlap_groups :
            if cam_a in g and cam_b in g :
                return True
        return False

    def _is_effectively_same_cam (self ,cam ,cams ,last_cam ):
        """A candidate is 'effectively same-camera' (NOT a cross-camera match)
        if THIS camera, or any overlapping camera, is among the cameras it was
        seen on. Overlapping cameras view the same room, so a person there is
        not physically 'somewhere else'."""
        if not cam :
            return False
        if cam in cams or self ._cams_overlap (cam ,last_cam ):
            return True
        for c in cams :
            if self ._cams_overlap (cam ,c ):
                return True
        return False

    def match (self ,vec ,cam =None ,
    threshold =0.55 ,soft_threshold =0.48 ,
    cross_camera_threshold =0.45 ,
    soft_window =120.0 ,margin =0.04 ,topk =3 ,
    exclude =None ,
    posture_threshold =None ,posture_window =12.0 ,
    cross_camera_floor =None ,cross_margin =0.12 ):
        # cross_camera_floor: a LOWER cross-camera bar that applies only when the
        # best cross-camera candidate beats the SECOND-best by cross_margin. The
        # fixed high cross-cam bar (0.86) was rejecting real hand-offs: the orange
        # man walking cam3->cam4 scored 0.846 against his own gallery — his true
        # match — but 0.846 < 0.86 so he was given a NEW UID. The problem is the
        # ReID model's stranger p95 (~0.84) sits right at the same-person cross-cam
        # score, so NO fixed bar separates them. A MARGIN does: a genuine match is
        # not only high but decisively higher than the next candidate, whereas an
        # accidental stranger collision has another stranger scoring nearly as
        # high. This recovers hand-offs without lowering stranger safety.
        # posture_threshold: a RELAXED same-camera bar that applies ONLY to a
        # candidate seen on this camera within posture_window seconds. A person
        # who stands then sits (or vice-versa) changes appearance enough to drop
        # ~0.04-0.10 below the normal same-cam bar (a standing full-body crop vs
        # a seated crop of the same person scores ~0.64 on ResNet50/Market1501),
        # so a cold 0.68 gate mints a NEW UID for them. But "seen here <12s ago
        # AND still the best match by a clear margin" is strong evidence it is
        # the same person mid-posture-change. The relaxation is same-camera and
        # recency-gated, so it does NOT lower the bar for strangers or cross-cam.

        vec =_l2 (vec )
        now =time .time ()
        exclude =exclude or set ()

        scored =[]
        for pid ,embs in self ._emb_cache .items ():
            if not embs or pid in exclude :
                continue

            mat =self ._mat_cache .get (pid )
            if mat is None or mat .shape [0 ]!=len (embs ):
                mat =np .asarray (embs ,dtype =np .float32 )
                self ._mat_cache [pid ]=mat
            sims =mat @vec
            k =min (topk ,sims .shape [0 ])
            top_mean =float (np .mean (np .sort (sims )[-k :]))
            meta =self ._meta_cache .get (pid ,{})
            recent =(now -meta .get ("last_seen",0 ))<soft_window
            cams =meta .get ("cams")or set ()
            last_cam =meta .get ("last_cam")
            # A candidate is CROSS-camera only if it was NOT seen on this camera
            # or any camera overlapping this one. Overlapping-FOV cameras (same
            # room) legitimately see the same person at once, so a candidate last
            # seen on an overlapping camera is treated as same-camera: it gets the
            # relaxed same-cam bar and the recency boost. Without this, cam3<->cam4
            # hand-offs were forced over the strict 0.84 cross bar and never fired
            # ("hand-offs succeeded: 0" in the logs) even though the two views are
            # of one room. A candidate on a truly separate camera stays cross.
            eff_same =self ._is_effectively_same_cam (cam ,cams ,last_cam )
            is_cross =bool (cam )and not eff_same
            # Recency boost applies to SAME-CAMERA and OVERLAPPING-camera
            # candidates (anti-flicker + in-room hand-off). It is still withheld
            # from TRUE cross-camera candidates, where inflating the score would
            # collapse the higher cross-cam bar and risk merging strangers.
            if recent and not is_cross :
                score =min (top_mean +SAME_CAM_RECENT_BOOST ,1.0 )
            else :
                score =top_mean
            # Very-recent same-camera flag for the posture-transition path. Also
            # fires for overlapping cameras, since a person moving between two
            # views of one room is the same posture-transition situation.
            _very_recent_here =(bool (cam )
            and (now -meta .get ("last_seen",0 ))<posture_window
            and eff_same )
            scored .append ((score ,pid ,recent ,is_cross ,_very_recent_here ))

        if not scored :
            return None ,0.0

        scored .sort (reverse =True )
        best_score ,best_pid ,best_recent ,best_cross ,best_very_recent =scored [0 ]
        second_score =scored [1 ][0 ]if len (scored )>1 else 0.0

        bar =cross_camera_threshold if best_cross else threshold

        # POSTURE-TRANSITION relaxation: seen on THIS camera very recently and
        # clearly the top match. Fires even for a multi-camera identity (one
        # already seen on an overlapping camera) — the decisive fact is that it
        # was on THIS physical camera seconds ago, which cross-camera-ness does
        # not change. Still gated by same-camera recency + a clear top margin,
        # so it cannot merge strangers.
        if (posture_threshold is not None
        and best_very_recent
        and (best_score -second_score )>=margin ):
            if best_score >=posture_threshold :
                return best_pid ,best_score

        confident =best_score >=(bar +0.10 )
        if confident :
            return best_pid ,best_score

        # MARGIN-GATED CROSS-CAMERA acceptance. For a cross-camera candidate that
        # clears cross_camera_floor AND beats a REAL runner-up by cross_margin,
        # accept it even though it is below the strict cross-cam bar. The decisive
        # gap is what distinguishes a real hand-off (one identity clearly closest)
        # from a stranger collision (several identities bunched near one score).
        #
        # CRITICAL: this requires a GENUINE competitor. With only one candidate,
        # second_score is 0.0, so "best - second" is trivially huge and a LONE
        # look-alike stranger sitting exactly at the floor (0.80) would sail
        # through — that is precisely how one global UID (SFK6AT) was painted on
        # three different seated men, each the only gallery hit on their camera.
        # A lone candidate carries no disambiguating evidence, so it must clear
        # the STRICT bar, not the floor. The floor relaxation applies only when a
        # real runner-up exists AND the best decisively beats it.
        MIN_REAL_RUNNERUP =min (soft_threshold ,cross_camera_floor -0.10 )
        has_real_runnerup =(len (scored )>=2 and second_score >=MIN_REAL_RUNNERUP )
        if (best_cross and cross_camera_floor is not None
        and has_real_runnerup
        and best_score >=cross_camera_floor
        and (best_score -second_score )>=cross_margin ):
            return best_pid ,best_score

        second_qualifies =second_score >=bar
        if (best_score -second_score )<margin and second_qualifies :
            return None ,best_score

        # The "soft" path allows a slightly lower bar for an identity seen very
        # recently. It must NEVER undercut the CROSS-CAMERA bar: with a 30-min
        # soft window almost every identity counts as "recent", which collapsed
        # the effective cross-cam threshold from 0.80 to soft_threshold (0.62)
        # and merged different people into one UID (observed stranger p95
        # climbing 0.67 -> 0.90 as identities got polluted).
        soft_ok =(best_score >=soft_threshold and best_recent
        and not best_cross )
        if best_score >=bar or soft_ok :
            return best_pid ,best_score

        return None ,best_score

    def _record_sighting (self ,pid ,cam ):

        if not cam :
            return
        now =time .time ()
        nowstr =_ts_str (now )
        cur =self ._conn .execute (
        "SELECT count FROM sightings WHERE person_id=? AND camera=?",
        (pid ,cam )).fetchone ()
        if cur is None :
            self ._conn .execute (
            "INSERT INTO sightings(person_id, camera, first_seen, last_seen, "
            "first_str, last_str, count) VALUES (?,?,?,?,?,?,1)",
            (pid ,cam ,now ,now ,nowstr ,nowstr ))
            print (f"[SIGHTING] {pid } first seen on {cam }")
        else :
            self ._conn .execute (
            "UPDATE sightings SET last_seen=?, last_str=?, count=count+1 "
            "WHERE person_id=? AND camera=?",(now ,nowstr ,pid ,cam ))

    def create_person (self ,vec ,cam =None )->str :
        vec =_l2 (vec )
        pid =self ._new_id ()
        now =time .time ()
        nowstr =_ts_str (now )
        self ._conn .execute (
        "INSERT INTO persons (person_id, created_at, last_seen, "
        "sighting_count, first_seen_str, last_seen_str, last_camera, "
        "num_embeddings) VALUES (?,?,?,?,?,?,?,?)",
        (pid ,now ,now ,1 ,nowstr ,nowstr ,cam ,1 ))
        self ._conn .execute (
        "INSERT INTO embeddings(person_id, vec, created_at, created_str, "
        "camera) VALUES (?,?,?,?,?)",
        (pid ,vec .astype (np .float32 ).tobytes (),now ,nowstr ,cam ))
        self ._conn .commit ()

        self ._emb_cache [pid ].append (vec )
        self ._mat_cache .pop (pid ,None )

        self ._meta_cache [pid ]={"last_seen":now ,"count":1 ,"last_cam":cam ,
        "cams":{cam }if cam else set (),
        "first_seen":now ,"first_seen_str":nowstr }
        self ._record_sighting (pid ,cam )
        print (f"[DB] NEW person {pid }  (total {len (self ._meta_cache )})")
        return pid

    def create_person_reidless (self ,cam =None )->str :
        """Create a global identity with NO embedding yet (Tier-1 local commit).
        Used when a track is confirmed but the ReID engine hasn't produced a
        vector. Embeddings are added later via add_embedding as they arrive."""
        pid =self ._new_id ()
        now =time .time ()
        nowstr =_ts_str (now )
        self ._conn .execute (
        "INSERT INTO persons (person_id, created_at, last_seen, "
        "sighting_count, first_seen_str, last_seen_str, last_camera, "
        "num_embeddings) VALUES (?,?,?,?,?,?,?,?)",
        (pid ,now ,now ,1 ,nowstr ,nowstr ,cam ,0 ))
        self ._conn .commit ()
        self ._emb_cache [pid ]=[]
        self ._meta_cache [pid ]={"last_seen":now ,"count":1 ,"last_cam":cam ,
        "cams":{cam }if cam else set (),
        "first_seen":now ,"first_seen_str":nowstr }
        self ._record_sighting (pid ,cam )
        print (f"[DB] NEW person {pid } (no-reid, total {len (self ._meta_cache )})")
        return pid

    def merge_persons (self ,keep_pid ,drop_pid ):
        """Merge drop_pid INTO keep_pid: move embeddings & sightings, keep the
        earliest first-seen. Returns True on success. Used for cross-camera
        hand-off when a post-commit ReID match reveals two UIDs are one person."""
        if keep_pid ==drop_pid :
            return False
        if keep_pid not in self ._meta_cache or drop_pid not in self ._meta_cache :
            return False
        km =self ._meta_cache [keep_pid ]
        dm =self ._meta_cache [drop_pid ]
        # Preserve the earliest identity as the survivor's first-seen.
        if dm .get ("first_seen")and (not km .get ("first_seen")
        or dm ["first_seen"]<km ["first_seen"]):
            km ["first_seen"]=dm ["first_seen"]
            km ["first_seen_str"]=dm .get ("first_seen_str")or _ts_str (dm ["first_seen"])
            self ._conn .execute (
            "UPDATE persons SET created_at=?, first_seen_str=? WHERE person_id=?",
            (km ["first_seen"],km ["first_seen_str"],keep_pid ))
        try :
            self ._conn .execute ("UPDATE embeddings SET person_id=? WHERE person_id=?",
            (keep_pid ,drop_pid ))
            # Sightings are UNIQUE(person_id, camera). A blind UPDATE collides
            # whenever BOTH identities have a row for the same camera (very common
            # for a cross-camera merge). Fold overlapping rows first: sum counts,
            # take the earliest first_seen and the latest last_seen, then delete
            # the drop rows. Only non-overlapping rows are reassigned.
            self ._conn .execute (
            "UPDATE sightings AS k SET "
            "  count = k.count + ("
            "     SELECT d.count FROM sightings d "
            "     WHERE d.person_id=? AND d.camera=k.camera), "
            "  first_seen = MIN(k.first_seen, ("
            "     SELECT d.first_seen FROM sightings d "
            "     WHERE d.person_id=? AND d.camera=k.camera)), "
            "  last_seen = MAX(k.last_seen, ("
            "     SELECT d.last_seen FROM sightings d "
            "     WHERE d.person_id=? AND d.camera=k.camera)) "
            "WHERE k.person_id=? AND EXISTS ("
            "     SELECT 1 FROM sightings d "
            "     WHERE d.person_id=? AND d.camera=k.camera)",
            (drop_pid ,drop_pid ,drop_pid ,keep_pid ,drop_pid ))
            # Remove the now-folded duplicate rows.
            self ._conn .execute (
            "DELETE FROM sightings WHERE person_id=? AND camera IN ("
            "   SELECT camera FROM sightings WHERE person_id=?)",
            (drop_pid ,keep_pid ))
            # Reassign whatever cameras remain unique to the dropped id.
            self ._conn .execute ("UPDATE sightings SET person_id=? WHERE person_id=?",
            (keep_pid ,drop_pid ))
            self ._conn .execute ("DELETE FROM persons WHERE person_id=?",(drop_pid ,))
            self ._retire_uid (drop_pid ,"merged")   # ledger keeps it forever
            self ._conn .commit ()
        except Exception as e :
            print (f"[DB] merge_persons({keep_pid },{drop_pid }) error: {e }")
            return False
        # Merge in-memory caches.
        merged =self ._emb_cache .get (keep_pid ,[])+self ._emb_cache .get (drop_pid ,[])
        if len (merged )>self .max_emb :
            merged =merged [-self .max_emb :]
        self ._emb_cache [keep_pid ]=merged
        self ._emb_cache .pop (drop_pid ,None )
        self ._mat_cache .pop (keep_pid ,None )
        self ._mat_cache .pop (drop_pid ,None )
        km ["cams"]=(km .get ("cams")or set ())|(dm .get ("cams")or set ())
        km ["count"]=km .get ("count",0 )+dm .get ("count",0 )
        km ["last_seen"]=max (km .get ("last_seen",0 ),dm .get ("last_seen",0 ))
        self ._meta_cache .pop (drop_pid ,None )
        self ._conn .execute (
        "UPDATE persons SET num_embeddings=? WHERE person_id=?",
        (len (merged ),keep_pid ))
        self ._conn .commit ()
        print (f"[DB] MERGED {drop_pid } -> {keep_pid } "
        f"(now {len (merged )} embeddings, cams={sorted (km ['cams'])})")
        return True

    def cams_of (self ,pid ):
        m =self ._meta_cache .get (pid )
        return set (m .get ("cams")or set ())if m else set ()

    def add_embedding (self ,pid ,vec ,cam =None ,outlier_floor =0.40 ,
    guard_contamination =True ):

        vec =_l2 (vec )

        if vec .shape [0 ]!=self .feat_sz or not np .all (np .isfinite (vec )):
            return False
        if float (np .linalg .norm (vec ))<1e-6 :
            return False
        embs =self ._emb_cache [pid ]
        if embs :
            avg =_l2 (np .mean (np .asarray (embs ,dtype =np .float32 ),axis =0 ))
            sim =self ._cosine (vec ,avg )

            redundant_bar =0.985 if len (embs )<self .seed_target else 0.93
            if sim >=redundant_bar :
                return False
            if sim <outlier_floor :
                return False

            # CONTAMINATION GUARD. A rising cross-identity p95 (different people
            # scoring ~0.83, at the cross-cam bar) means UIDs are absorbing other
            # people's crops. Refuse a crop that matches SOME OTHER identity's
            # centroid better than it matches its own (by a clear margin): that
            # crop almost certainly belongs to the other person, and adding it
            # would pull this UID's centroid toward theirs, blurring the two.
            if guard_contamination :
                best_other =0.0
                for opid ,oembs in self ._emb_cache .items ():
                    if opid ==pid or not oembs :
                        continue
                    om =self ._mat_cache .get (opid )
                    if om is None or om .shape [0 ]!=len (oembs ):
                        om =np .asarray (oembs ,dtype =np .float32 )
                        self ._mat_cache [opid ]=om
                    os_ =float (np .max (om @vec ))
                    if os_ >best_other :
                        best_other =os_
                if best_other >sim +CONTAM_MARGIN :
                    return False

        now =time .time ()
        self ._conn .execute (
        "INSERT INTO embeddings(person_id, vec, created_at, created_str, "
        "camera) VALUES (?,?,?,?,?)",
        (pid ,vec .astype (np .float32 ).tobytes (),now ,_ts_str (now ),cam ))
        self ._conn .execute ("""
            DELETE FROM embeddings WHERE id IN (
                SELECT id FROM embeddings WHERE person_id=?
                ORDER BY created_at DESC LIMIT -1 OFFSET ?
            )""",(pid ,self .max_emb ))

        embs .append (vec )
        if len (embs )>self .max_emb :
            del embs [0 ]

        self ._mat_cache .pop (pid ,None )

        self ._conn .execute (
        "UPDATE persons SET num_embeddings=? WHERE person_id=?",
        (len (embs ),pid ))
        self ._conn .commit ()
        return True

    # ---- Fixture (non-person) appearance prototypes -----------------------
    def learn_fixture (self ,vec ,cam =None ):
        """Record the appearance of a confirmed non-person (empty chair / bike
        row). Near-duplicate prototypes are folded together with a running
        average so the gallery stays small and each real fixture is one entry."""
        if vec is None :
            return
        vec =_l2 (np .asarray (vec ,dtype =np .float32 ))
        if vec .shape [0 ]!=self .feat_sz or not np .all (np .isfinite (vec )):
            return
        now =time .time ()
        for p in self ._fixture_protos :
            if p ["cam"]==cam and self ._cosine (vec ,p ["vec"])>=self ._fixture_merge_thr :
                n =p ["n"]
                p ["vec"]=_l2 ((p ["vec"]*n +vec )/(n +1 ))
                p ["n"]=n +1
                p ["ts"]=now
                return
        self ._fixture_protos .append ({"vec":vec ,"cam":cam ,"ts":now ,"n":1 })
        if len (self ._fixture_protos )>self ._fixture_max :
            self ._fixture_protos .sort (key =lambda p :p ["ts"])
            self ._fixture_protos =self ._fixture_protos [-self ._fixture_max :]

    def is_fixture_appearance (self ,vec ,cam =None ):
        """True if `vec` looks like a known fixture on this camera. Prototypes
        expire after _fixture_ttl so a chair a person now occupies is released.
        Matching is same-camera only: a chair on cam-04 must not veto a real
        person on cam-08."""
        if vec is None or not self ._fixture_protos :
            return False ,0.0
        vec =_l2 (np .asarray (vec ,dtype =np .float32 ))
        if vec .shape [0 ]!=self .feat_sz or not np .all (np .isfinite (vec )):
            return False ,0.0
        now =time .time ()
        self ._fixture_protos =[p for p in self ._fixture_protos
        if now -p ["ts"]<=self ._fixture_ttl ]
        best =0.0
        for p in self ._fixture_protos :
            if cam is not None and p ["cam"]is not None and p ["cam"]!=cam :
                continue
            s =self ._cosine (vec ,p ["vec"])
            if s >best :
                best =s
        return (best >=self ._fixture_reject_thr ),best

    def touch (self ,pid ,cam =None ):

        now =time .time ()
        m =self ._meta_cache .get (pid )
        if m :
            m ["last_seen"]=now
            m ["count"]+=1
            if cam is not None :
                m ["last_cam"]=cam
                m .setdefault ("cams",set ()).add (cam )
        self ._record_sighting (pid ,cam )
        self ._conn .execute (
        "UPDATE persons SET last_seen=?, last_seen_str=?, "
        "sighting_count=sighting_count+1, "
        "last_camera=COALESCE(?, last_camera) WHERE person_id=?",
        (now ,_ts_str (now ),cam ,pid ))
        self ._dirty =True
        if now -self ._last_commit >self .commit_interval :
            self ._conn .commit ()
            self ._dirty =False
            self ._last_commit =now

    def _record_sighting (self ,pid ,cam ):

        if not cam :
            return
        now =time .time ()
        nowstr =_ts_str (now )

        self ._conn .execute ("""
            INSERT INTO sightings (person_id, camera, first_seen, last_seen,
                                   first_str, last_str, count)
            VALUES (?,?,?,?,?,?,1)
            ON CONFLICT(person_id, camera) DO UPDATE SET
                last_seen=excluded.last_seen,
                last_str =excluded.last_str,
                count    =count+1
        """,(pid ,cam ,now ,now ,nowstr ,nowstr ))

    def gallery_size (self ,pid ):

        return len (self ._emb_cache .get (pid ,()))

    def first_seen_of (self ,pid ):

        m =self ._meta_cache .get (pid )
        if m and m .get ("first_seen"):
            return m ["first_seen"],m .get ("first_seen_str")or _ts_str (m ["first_seen"])
        row =self ._conn .execute (
        "SELECT created_at, first_seen_str FROM persons WHERE person_id=?",
        (pid ,)).fetchone ()
        if row :
            ts ,s =row [0 ],row [1 ]or _ts_str (row [0 ])
            return ts ,s
        return None ,""

    def cameras_for (self ,pid ):

        return [r [0 ]for r in self ._conn .execute (
        "SELECT camera FROM sightings WHERE person_id=? ORDER BY last_seen DESC",
        (pid ,))]

    def delete_person (self ,pid ):

        try :
            self ._conn .execute ("DELETE FROM embeddings WHERE person_id=?",(pid ,))
            self ._conn .execute ("DELETE FROM sightings  WHERE person_id=?",(pid ,))
            self ._conn .execute ("DELETE FROM persons     WHERE person_id=?",(pid ,))
            self ._retire_uid (pid ,"pruned")        # ledger keeps it forever
            self ._conn .commit ()
        except Exception as e :
            print (f"[DB] delete_person({pid }) DB error: {e }")
        self ._emb_cache .pop (pid ,None )
        self ._mat_cache .pop (pid ,None )
        self ._meta_cache .pop (pid ,None )
        print (f"[DB] DELETED phantom person {pid }  (total {len (self ._meta_cache )})")

    def close (self ):
        if self ._dirty :
            self ._conn .commit ()
        self ._conn .commit ()
        self ._conn .close ()

    def stats (self ):
        return {
        "persons":len (self ._meta_cache ),
        "total_embeddings":sum (len (v )for v in self ._emb_cache .values ()),
        }

if __name__ =="__main__":
    import tempfile ,os
    tmp =tempfile .mktemp (suffix =".db")
    db =PersonDatabase (db_path =tmp ,feature_size =8 )
    rng =np .random .default_rng (0 )
    a =rng .standard_normal (8 ).astype (np .float32 )
    pid_a =db .create_person (a )

    a2 =a +0.01 *rng .standard_normal (8 ).astype (np .float32 )
    m ,s =db .match (_l2 (a2 ),threshold =0.70 )
    assert m ==pid_a ,f"expected re-id, got {m } score {s :.3f}"

    b =rng .standard_normal (8 ).astype (np .float32 )
    m2 ,s2 =db .match (_l2 (b ),threshold =0.70 )
    print (f"self-test OK: reid score={s :.3f}, stranger score={s2 :.3f}")
    db .close ()
    os .remove (tmp )
