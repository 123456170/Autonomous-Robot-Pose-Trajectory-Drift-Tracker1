import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import json
import hashlib
import sqlite3
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
import plotly.express as px
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ======================================================================================================================
# STREAMLIT PAGE CONFIG
# ======================================================================================================================
st.set_page_config(
    page_title="Autonomous Robot Pose & Trajectory Drift Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Autonomous Robot Pose & Trajectory Drift Tracker"
APP_SUBTITLE = (
    "Local-first synthetic robotics workcell: pose tracking, trajectory drift, AI detection, "
    "investigation, risk, policy-gated simulator-only action, independent verification, evidence, and replay."
)

WIDTH = 960
HEIGHT = 540
DEMO_DURATION_S = 120
DB_PATH = "pose_drift_tracker.db"
CONFIG_VERSION = "cfg_v1"

# ======================================================================================================================
# SMALL HELPERS
# ======================================================================================================================
def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def norm_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def point_in_rect(x: float, y: float, rect: Optional[Tuple[int, int, int, int] | Dict[str, float]]) -> bool:
    if rect is None:
        return False
    if isinstance(rect, dict):
        rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
    else:
        rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def demo_phase(t: float) -> str:
    if t < 20:
        return "0-20s Normal workcell"
    if t < 45:
        return "20-45s Unique anomaly injected: R2 coordinate drift"
    if t < 70:
        return "45-70s Persistent tracking through occlusion"
    if t < 90:
        return "70-90s AI detection & explanation"
    if t < 105:
        return "90-105s Investigation & risk"
    if t < 115:
        return "105-115s Policy-gated simulated action"
    return "115-120s Verification, evidence & replay"


# ======================================================================================================================
# PYDANTIC DATA CONTRACTS
# ======================================================================================================================
class CameraFrame(BaseModel):
    frame_id: str
    ts: float
    source: str = "cam0"
    width: int = WIDTH
    height: int = HEIGHT
    hash: str = ""
    provenance: str = "synthetic"
    version: int = 1


class SensorObservation(BaseModel):
    obs_id: str
    frame_id: str
    entity_id: str
    x: float
    y: float
    heading: float
    confidence: float
    source: str = "synthetic-camera"
    ts: float = 0.0
    provenance: str = "simulator"


class EntityTrack(BaseModel):
    track_id: str
    entity_id: str
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.8
    active: bool = True
    miss_count: int = 0
    history_len: int = 0
    provenance: str = "tracker"


class PoseState(BaseModel):
    pose_id: str
    frame_id: str
    entity_id: str
    x: float
    y: float
    heading: float
    confidence: float
    source: str = "state_estimator"
    ts: float = 0.0


class Relationship(BaseModel):
    rel_id: str
    frame_id: str
    subject: str
    predicate: str
    object_: str = Field(alias="object")
    confidence: float
    ts: float = 0.0


class Detection(BaseModel):
    detection_id: str
    frame_id: str
    entity_id: str
    bbox: List[float]
    confidence: float
    ts: float = 0.0


class Hypothesis(BaseModel):
    hypothesis_id: str
    investigation_id: str
    text: str
    score: float
    ts: float = 0.0


class Investigation(BaseModel):
    investigation_id: str
    entity_id: str
    status: str
    opened_at: float
    closed_at: Optional[float] = None
    hypotheses: List[str] = []
    provenance: str = "correlation_engine"


class RiskScore(BaseModel):
    entity_id: str
    score: float
    anomaly: float
    criticality: float
    persistence: float
    confidence: float
    evidence: float
    impact: float
    ts: float = 0.0


class ProposedAction(BaseModel):
    action_id: str
    frame_id: str
    entity_id: str
    kind: str
    policy: str
    risk: float
    before_error: float
    ts: float = 0.0
    simulation_only: bool = True


class Verification(BaseModel):
    verification_id: str
    action_id: str
    frame_id: str
    success: bool
    before_error: float
    after_error: float
    confidence: float
    latency_s: float
    ts: float = 0.0


class Evidence(BaseModel):
    evidence_id: str
    frame_id: str
    action_id: str
    verification_id: str
    hash: str
    payload: Dict[str, Any]
    ts: float = 0.0


class AuditEvent(BaseModel):
    event_id: str
    ts: float
    kind: str
    payload: Dict[str, Any]


class Metric(BaseModel):
    metric_id: str
    frame_id: str
    ts: float
    payload: Dict[str, Any]


# ======================================================================================================================
# SQLITE
# ======================================================================================================================
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames(
    frame_id TEXT PRIMARY KEY,
    ts REAL,
    source TEXT,
    width INTEGER,
    height INTEGER,
    hash TEXT,
    provenance TEXT,
    version INTEGER
);

CREATE TABLE IF NOT EXISTS observations(
    obs_id TEXT PRIMARY KEY,
    frame_id TEXT,
    entity_id TEXT,
    x REAL,
    y REAL,
    heading REAL,
    confidence REAL,
    source TEXT,
    ts REAL,
    provenance TEXT
);

CREATE TABLE IF NOT EXISTS tracks(
    track_record_id TEXT PRIMARY KEY,
    frame_id TEXT,
    track_id TEXT,
    entity_id TEXT,
    x REAL,
    y REAL,
    heading REAL,
    vx REAL,
    vy REAL,
    confidence REAL,
    active INTEGER,
    miss INTEGER,
    ts REAL
);

CREATE TABLE IF NOT EXISTS poses(
    pose_id TEXT PRIMARY KEY,
    frame_id TEXT,
    entity_id TEXT,
    x REAL,
    y REAL,
    heading REAL,
    confidence REAL,
    source TEXT,
    ts REAL
);

CREATE TABLE IF NOT EXISTS relationships(
    rel_id TEXT PRIMARY KEY,
    frame_id TEXT,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence REAL,
    ts REAL
);

CREATE TABLE IF NOT EXISTS detections(
    detection_id TEXT PRIMARY KEY,
    frame_id TEXT,
    entity_id TEXT,
    bbox TEXT,
    confidence REAL,
    ts REAL
);

CREATE TABLE IF NOT EXISTS investigations(
    investigation_id TEXT PRIMARY KEY,
    entity_id TEXT,
    status TEXT,
    opened_at REAL,
    closed_at REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses(
    hypothesis_id TEXT PRIMARY KEY,
    investigation_id TEXT,
    text TEXT,
    score REAL,
    ts REAL
);

CREATE TABLE IF NOT EXISTS actions(
    action_id TEXT PRIMARY KEY,
    frame_id TEXT,
    entity_id TEXT,
    kind TEXT,
    policy TEXT,
    risk REAL,
    before_error REAL,
    ts REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS verifications(
    verification_id TEXT PRIMARY KEY,
    action_id TEXT,
    frame_id TEXT,
    success INTEGER,
    before_error REAL,
    after_error REAL,
    confidence REAL,
    ts REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS evidence(
    evidence_id TEXT PRIMARY KEY,
    frame_id TEXT,
    action_id TEXT,
    verification_id TEXT,
    hash TEXT,
    payload TEXT,
    ts REAL
);

CREATE TABLE IF NOT EXISTS model_runs(
    run_id TEXT PRIMARY KEY,
    frame_id TEXT,
    model_version TEXT,
    config_version TEXT,
    payload TEXT,
    ts REAL
);

CREATE TABLE IF NOT EXISTS metrics(
    metric_id TEXT PRIMARY KEY,
    frame_id TEXT,
    ts REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS audit_events(
    event_id TEXT PRIMARY KEY,
    ts REAL,
    kind TEXT,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_obs_frame ON observations(frame_id);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_tracks_frame ON tracks(frame_id);
CREATE INDEX IF NOT EXISTS idx_poses_frame ON poses(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_metrics_frame ON metrics(frame_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts);
"""

_db_lock = threading.Lock()


@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


def db_write_many(statements: List[Tuple[str, tuple]]):
    if not st.session_state.get("db_enabled", True) or not statements:
        return
    try:
        with _db_lock:
            conn = get_db()
            for sql, params in statements:
                conn.execute(sql, params)
            conn.commit()
    except Exception as e:
        st.session_state.db_error = str(e)


# ======================================================================================================================
# SYNTHETIC WORKCELL DEFINITION
# ======================================================================================================================
ROBOTS = [
    {"id": "R1", "color": (0, 200, 255), "role": "AGV"},
    {"id": "R2", "color": (0, 255, 120), "role": "Mobile manipulator"},
    {"id": "R3", "color": (0, 120, 255), "role": "Inspection robot"},
]
ROBOT_IDS = [r["id"] for r in ROBOTS]
ROBOT_MAP = {r["id"]: r for r in ROBOTS}

ZONES = [
    {"name": "Assembly A", "x": 80, "y": 80, "w": 260, "h": 190, "critical": 0.9},
    {"name": "Human Walkway", "x": 420, "y": 330, "w": 300, "h": 150, "critical": 1.0},
    {"name": "Charging", "x": 730, "y": 60, "w": 180, "h": 140, "critical": 0.6},
    {"name": "Buffer", "x": 120, "y": 350, "w": 220, "h": 140, "critical": 0.4},
]


def true_pose(entity_id: str, t: float) -> Dict[str, float]:
    if entity_id == "R1":
        cx, cy = 480, 250
        rx, ry = 280, 150
        w = 0.35
        ang = 0.7 + w * t
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        heading = norm_angle(ang + math.pi / 2)
    elif entity_id == "R2":
        span = 560.0
        minx = 180.0
        speed = 65.0
        phase = (t * speed) % (2 * span)
        if phase < span:
            x = minx + phase
            heading = 0.0
        else:
            x = minx + 2 * span - phase
            heading = math.pi
        y = 390.0
    else:
        cx, cy = 760, 140
        rx, ry = 90, 70
        w = 0.6
        ang = 2.1 + w * t
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        heading = norm_angle(ang + math.pi / 2)

    return {
        "x": float(clamp(x, 15, WIDTH - 15)),
        "y": float(clamp(y, 15, HEIGHT - 15)),
        "heading": float(norm_angle(heading)),
    }


def get_occluder(t: float) -> Optional[Tuple[int, int, int, int]]:
    if 45.0 <= t < 70.0:
        x = int(150 + ((t - 45.0) / 25.0) * 600)
        return (x, 160, 120, 260)
    return None


def zone_critical(x: float, y: float) -> float:
    crit = 0.15
    for z in ZONES:
        if point_in_rect(x, y, z):
            crit = max(crit, float(z["critical"]))
    return crit


# ======================================================================================================================
# TRACKING ENGINE
# ======================================================================================================================
@dataclass
class Track:
    track_id: int
    entity_id: str
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    last_frame: int = 0
    miss: int = 0
    active: bool = True
    conf: float = 0.8
    color: Tuple[int, int, int] = (200, 200, 200)
    history: List[Tuple[int, float, float]] = field(default_factory=list)


class TrackingEngine:
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.id_switches = 0
        self.entity_to_track: Dict[str, int] = {}
        self.total_updates = 0
        self.associated_obs = 0

    def active(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.active]

    def _register_entity_switch(self, true_id: str, track_id: int):
        prev = self.entity_to_track.get(true_id)
        if prev is not None and prev != track_id:
            self.id_switches += 1
        self.entity_to_track[true_id] = track_id

    def _update_track(self, t: Track, obs: Dict[str, Any], frame_idx: int, fps: int):
        prev_frame = t.last_frame
        dt_obs = max(1.0 / max(1, fps), (frame_idx - prev_frame) / float(fps) if frame_idx > prev_frame else 1.0 / max(1, fps))
        alpha = 0.55

        if frame_idx > prev_frame:
            nvx = (obs["x"] - t.x) / dt_obs
            nvy = (obs["y"] - t.y) / dt_obs
            t.vx = alpha * nvx + (1 - alpha) * t.vx
            t.vy = alpha * nvy + (1 - alpha) * t.vy

        hx = (1 - alpha) * math.cos(t.heading) + alpha * math.cos(obs["heading"])
        hy = (1 - alpha) * math.sin(t.heading) + alpha * math.sin(obs["heading"])
        t.heading = norm_angle(math.atan2(hy, hx))

        t.x = alpha * obs["x"] + (1 - alpha) * t.x
        t.y = alpha * obs["y"] + (1 - alpha) * t.y
        t.conf = alpha * obs["confidence"] + (1 - alpha) * t.conf
        t.color = tuple(
            int(alpha * c1 + (1 - alpha) * c2)
            for c1, c2 in zip(obs["appearance"], t.color)
        )
        t.last_frame = frame_idx
        t.miss = 0
        t.active = True
        t.history.append((frame_idx, t.x, t.y))
        if len(t.history) > 240:
            t.history.pop(0)

        self._register_entity_switch(obs["true_id"], t.track_id)

    def update(self, observations: List[Dict[str, Any]], frame_idx: int, fps: int) -> List[Track]:
        dt = 1.0 / max(1, fps)
        active = self.active()

        for t in active:
            t.px = t.x + t.vx * dt
            t.py = t.y + t.vy * dt

        costs = []
        for oi, obs in enumerate(observations):
            for t in active:
                dist = math.hypot(obs["x"] - getattr(t, "px", t.x), obs["y"] - getattr(t, "py", t.y))
                cdist = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs["appearance"], t.color))) / 441.0
                cost = dist + 120.0 * cdist
                if cost < 150.0:
                    costs.append((cost, oi, t.track_id))

        costs.sort(key=lambda z: z[0])
        assigned_obs = set()
        assigned_trk = set()

        for cost, oi, tid in costs:
            if oi in assigned_obs or tid in assigned_trk:
                continue
            assigned_obs.add(oi)
            assigned_trk.add(tid)
            self._update_track(self.tracks[tid], observations[oi], frame_idx, fps)
            self.associated_obs += 1
            self.total_updates += 1

        max_miss = int(2.5 * fps)
        for t in active:
            if t.track_id not in assigned_trk:
                t.miss += 1
                t.conf = max(0.05, t.conf * 0.93)
                if t.miss > max_miss:
                    t.active = False

        for oi, obs in enumerate(observations):
            if oi in assigned_obs:
                continue

            candidate = None
            for t in self.tracks.values():
                if (
                    not t.active
                    and t.entity_id == obs["entity_id"]
                    and (frame_idx - t.last_frame) < int(6 * fps)
                    and math.hypot(obs["x"] - t.x, obs["y"] - t.y) < 170
                ):
                    candidate = t
                    break

            if candidate is not None:
                candidate.active = True
                candidate.miss = 0
                self._update_track(candidate, obs, frame_idx, fps)
                self.total_updates += 1
            elif obs["confidence"] > 0.35:
                tid = self.next_track_id
                self.next_track_id += 1
                trk = Track(
                    track_id=tid,
                    entity_id=obs["entity_id"],
                    x=obs["x"],
                    y=obs["y"],
                    heading=obs["heading"],
                    last_frame=frame_idx,
                    conf=obs["confidence"],
                    color=tuple(int(c) for c in obs["appearance"]),
                    history=[(frame_idx, obs["x"], obs["y"])],
                )
                self.tracks[tid] = trk
                self._register_entity_switch(obs["true_id"], tid)
                self.total_updates += 1

        return self.active()


# ======================================================================================================================
# AI ENGINE
# ======================================================================================================================
class AIDetector:
    def __init__(self):
        self.version = "pose-ai-v1"
        self.stats: Dict[str, Dict[str, float]] = {}
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        sim: "Simulator",
        tracks: List[Track],
        true_poses: Dict[str, Dict[str, float]],
        obs_by_entity: Dict[str, Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        track_by_entity = {t.entity_id: t for t in tracks if t.entity_id}
        results = {}

        for rid in ROBOT_IDS:
            trk = track_by_entity.get(rid)
            if trk is None:
                results[rid] = {
                    "score": 0.0,
                    "det": False,
                    "baseline": False,
                    "residual": 0.0,
                    "heading_err": 0.0,
                    "explanation": ["No active track"],
                }
                continue

            truth = true_poses[rid]
            residual = math.hypot(trk.x - truth["x"], trk.y - truth["y"])
            heading_err = abs(norm_angle(trk.heading - truth["heading"]))

            stat = self.stats.setdefault(rid, {"mean": 5.0, "m2": 25.0, "n": 5})
            if (sim.time < 18.0 and residual < 20.0) or residual < 9.0:
                n = stat["n"] + 1
                delta = residual - stat["mean"]
                stat["mean"] += delta / n
                stat["m2"] += delta * (residual - stat["mean"])
                stat["n"] = n

            var = stat["m2"] / max(1, stat["n"] - 1)
            std = max(2.0, math.sqrt(max(4.0, var)))
            z = (residual - stat["mean"]) / std
            hz = heading_err / 0.15
            conf = trk.conf

            raw = (
                0.95 * max(0.0, z - 2.1)
                + 0.35 * max(0.0, hz - 1.0)
                + 0.8 * max(0.0, 0.65 - conf)
            )
            score = 1.0 / (1.0 + math.exp(-(raw - 0.8)))

            # Demo timing: keep official AI detection/explanation aligned with the 70-90s phase.
            if sim.time < 62.0:
                score = min(score, 0.28)

            score = clamp(score)
            baseline = residual > max(18.0, stat["mean"] + 3.0 * std)
            det = bool(score > 0.65 and residual > 16.0)

            explanation = []
            if residual > stat["mean"] + 2 * std:
                explanation.append(f"Position residual {residual:.1f}px ({z:.1f} sigma above normal)")
            if heading_err > 0.15:
                explanation.append(f"Heading error {heading_err:.2f} rad")
            if conf < 0.65:
                explanation.append(f"Low track confidence {conf:.2f}")
            if det and not explanation:
                explanation.append("Persistent deviation from planned trajectory")
            if not explanation:
                explanation.append("No dominant anomaly factor")

            results[rid] = {
                "score": float(score),
                "det": bool(det),
                "baseline": bool(baseline),
                "residual": float(residual),
                "heading_err": float(heading_err),
                "explanation": explanation,
            }

            self.history.append(
                {
                    "t": sim.time,
                    "rid": rid,
                    "score": float(score),
                    "residual": float(residual),
                }
            )

        if len(self.history) > 6000:
            self.history = self.history[-6000:]

        return results


# ======================================================================================================================
# SIMULATOR
# ======================================================================================================================
class Simulator:
    def __init__(self, fps: int = 6):
        self.fps = max(2, int(fps))
        self.reset()

    @property
    def time(self) -> float:
        return self.frame_idx / float(max(1, self.fps))

    def reset(self):
        self.frame_idx = 0
        self.tracker = TrackingEngine()
        self.ai = AIDetector()

        self.investigations: List[Dict[str, Any]] = []
        self.open_investigations: Dict[str, str] = {}
        self.action_log: List[Dict[str, Any]] = []
        self.verifications: List[Dict[str, Any]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.audit: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []
        self.eval_records: List[Dict[str, Any]] = []

        self.bias = {rid: (0.0, 0.0, 0.0) for rid in ROBOT_IDS}
        self.drift_corrected = False
        self.active_action: Optional[Dict[str, Any]] = None

        self.detection_streak = {rid: 0 for rid in ROBOT_IDS}
        self.risk_scores = {rid: 0.0 for rid in ROBOT_IDS}
        self.last_risk_components: Dict[str, Dict[str, float]] = {}

        self.policy_decision = "deny"
        self.last_policy_reason = "Initializing"
        self.last_processing_ms = 0.0
        self.last_frame_data: Dict[str, Any] = {}

        self.action_count = 0
        self.verification_count = 0
        self.successful_actions = 0
        self.total_detections = 0

        self.process_current()

    def add_audit(self, kind: str, payload: Dict[str, Any], statements: List[Tuple[str, tuple]]):
        event_id = f"evt_{self.frame_idx:06d}_{kind.replace(' ', '_')}_{len(self.audit)}"
        ts = time.time()
        self.audit.append(
            {
                "event_id": event_id,
                "ts": ts,
                "kind": kind,
                "payload": payload,
            }
        )
        if len(self.audit) > 800:
            self.audit = self.audit[-800:]
        statements.append(
            (
                "INSERT OR REPLACE INTO audit_events VALUES (?,?,?,?)",
                (event_id, ts, kind, canonical(payload)),
            )
        )

    def update_drift(self, t: float, cfg: Dict[str, Any]):
        if cfg["scenario"] == "normal":
            self.bias = {rid: (0.0, 0.0, 0.0) for rid in ROBOT_IDS}
            return

        strength = float(cfg["anomaly_strength"])
        if cfg["scenario"] == "high":
            strength *= 1.5

        rid = "R2"
        if t >= 20.0 and not self.drift_corrected:
            k = clamp((t - 20.0) / 25.0) * strength
            self.bias[rid] = (38.0 * k, 24.0 * k, 0.30 * k)
        elif self.drift_corrected:
            bx, by, bh = self.bias[rid]
            self.bias[rid] = (bx * 0.45, by * 0.45, bh * 0.45)

        for other in ROBOT_IDS:
            if other != rid:
                self.bias[other] = (0.0, 0.0, 0.0)

    def generate_observations(
        self,
        frame_id: str,
        t: float,
        true_poses: Dict[str, Dict[str, float]],
        occluder: Optional[Tuple[int, int, int, int]],
        cfg: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        observations = []
        detections = []
        missed = 0

        for i, robot in enumerate(ROBOTS):
            rid = robot["id"]
            truth = true_poses[rid]
            bx, by, bh = self.bias[rid]

            rng = np.random.default_rng(9000 + self.frame_idx * 17 + i * 31)
            noise_scale = float(cfg["noise"]) + (0.4 if cfg["scenario"] == "high" else 0.0)

            x = truth["x"] + bx + float(rng.normal(0, noise_scale))
            y = truth["y"] + by + float(rng.normal(0, noise_scale))
            heading = norm_angle(truth["heading"] + bh + float(rng.normal(0, 0.02)))

            conf = 0.93
            if occluder and point_in_rect(truth["x"], truth["y"], occluder):
                conf -= 0.48
            if t < 4.0:
                conf -= 0.18
            conf = clamp(conf, 0.05, 0.98)

            is_missed = False
            if occluder and point_in_rect(truth["x"], truth["y"], occluder) and (self.frame_idx % 3 != 0):
                is_missed = True
            if conf < 0.30:
                is_missed = True

            if is_missed:
                missed += 1
                continue

            obs_id = f"obs_{self.frame_idx:06d}_{rid}"
            obs = {
                "obs_id": obs_id,
                "frame_id": frame_id,
                "entity_id": rid,
                "true_id": rid,
                "x": float(x),
                "y": float(y),
                "heading": float(heading),
                "confidence": float(conf),
                "appearance": robot["color"],
                "source": "synthetic-camera",
                "truth": truth,
            }
            observations.append(obs)

            detections.append(
                {
                    "detection_id": f"det_{self.frame_idx:06d}_{rid}",
                    "frame_id": frame_id,
                    "entity_id": rid,
                    "bbox": [float(x - 22), float(y - 22), 44.0, 44.0],
                    "confidence": float(conf),
                }
            )

        return observations, detections, missed

    def advance(self):
        total = int(DEMO_DURATION_S * self.fps)
        if self.frame_idx + 1 >= total:
            if st.session_state.get("loop_demo", True):
                self.reset()
                return
            st.session_state.running = False
            return

        self.frame_idx += 1
        self.process_current()

    def process_current(self):
        start = time.perf_counter()
        cfg = current_config()
        self.fps = max(2, int(cfg["fps"]))

        wall_ts = time.time()
        t = self.time
        frame_id = f"frm_{self.frame_idx:06d}"
        frame_hash = sha256_of(
            {
                "frame_id": frame_id,
                "t": round(t, 3),
                "seed": self.frame_idx,
                "config_version": CONFIG_VERSION,
            }
        )

        db_statements: List[Tuple[str, tuple]] = []
        db_statements.append(
            (
                "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?)",
                (frame_id, wall_ts, "cam0", WIDTH, HEIGHT, frame_hash, "synthetic", 1),
            )
        )

        self.update_drift(t, cfg)
        true_poses = {rid: true_pose(rid, t) for rid in ROBOT_IDS}
        occluder = get_occluder(t)

        observations, detections, missed_obs = self.generate_observations(
            frame_id, t, true_poses, occluder, cfg
        )

        for obs in observations:
            db_statements.append(
                (
                    "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        obs["obs_id"],
                        frame_id,
                        obs["entity_id"],
                        obs["x"],
                        obs["y"],
                        obs["heading"],
                        obs["confidence"],
                        obs["source"],
                        wall_ts,
                        "synthetic",
                    ),
                )
            )

        for det in detections:
            db_statements.append(
                (
                    "INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?)",
                    (
                        det["detection_id"],
                        frame_id,
                        det["entity_id"],
                        json.dumps(det["bbox"]),
                        det["confidence"],
                        wall_ts,
                    ),
                )
            )

        active_tracks = self.tracker.update(observations, self.frame_idx, self.fps)

        for trk in active_tracks:
            track_rec_id = f"trkrec_{self.frame_idx:06d}_{trk.track_id}"
            db_statements.append(
                (
                    "INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        track_rec_id,
                        frame_id,
                        f"T{trk.track_id:03d}",
                        trk.entity_id,
                        trk.x,
                        trk.y,
                        trk.heading,
                        trk.vx,
                        trk.vy,
                        trk.conf,
                        int(trk.active),
                        trk.miss,
                        wall_ts,
                    ),
                )
            )

            pose_id = f"pose_{self.frame_idx:06d}_{trk.entity_id}"
            db_statements.append(
                (
                    "INSERT OR REPLACE INTO poses VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        pose_id,
                        frame_id,
                        trk.entity_id,
                        trk.x,
                        trk.y,
                        trk.heading,
                        trk.conf,
                        "tracker",
                        wall_ts,
                    ),
                )
            )

        obs_by_entity = {o["entity_id"]: o for o in observations}
        ai_results = self.ai.update(self, active_tracks, true_poses, obs_by_entity, cfg)

        if self.frame_idx % 3 == 0:
            payload = {
                rid: {
                    "score": ai_results[rid]["score"],
                    "det": ai_results[rid]["det"],
                    "baseline": ai_results[rid]["baseline"],
                }
                for rid in ROBOT_IDS
            }
            db_statements.append(
                (
                    "INSERT OR REPLACE INTO model_runs VALUES (?,?,?,?,?,?)",
                    (
                        f"run_{self.frame_idx:06d}",
                        frame_id,
                        self.ai.version,
                        CONFIG_VERSION,
                        canonical(payload),
                        wall_ts,
                    ),
                )
            )

        for rid in ROBOT_IDS:
            if ai_results[rid]["det"]:
                self.detection_streak[rid] += 1
                self.total_detections += 1
            else:
                self.detection_streak[rid] = 0

            label = int(
                rid == "R2"
                and t >= 65.0
                and (not self.drift_corrected)
                and math.hypot(self.bias[rid][0], self.bias[rid][1]) > 8.0
            )

            self.eval_records.append(
                {
                    "t": t,
                    "rid": rid,
                    "label": label,
                    "ai_pred": int(ai_results[rid]["det"]),
                    "baseline_pred": int(ai_results[rid]["baseline"]),
                    "ai_score": float(ai_results[rid]["score"]),
                }
            )

        track_by_entity = {tr.entity_id: tr for tr in active_tracks}

        # Investigation / duplicate prevention
        for rid in ROBOT_IDS:
            if self.detection_streak[rid] == 4 and rid not in self.open_investigations:
                inv_id = f"inv_{self.frame_idx:06d}_{rid}"
                inv = {
                    "investigation_id": inv_id,
                    "entity_id": rid,
                    "status": "open",
                    "opened_at": t,
                    "opened_frame": self.frame_idx,
                    "hypotheses": [
                        "Coordinate frame drift",
                        "Odometry bias",
                        "Marker occlusion / low confidence",
                    ],
                    "timeline": [
                        f"t={t:.1f}s detection streak reached 4",
                        f"AI score={ai_results[rid]['score']:.2f}",
                        f"residual={ai_results[rid]['residual']:.1f}px",
                    ],
                }
                self.investigations.append(inv)
                self.open_investigations[rid] = inv_id

                db_statements.append(
                    (
                        "INSERT OR REPLACE INTO investigations VALUES (?,?,?,?,?,?)",
                        (inv_id, rid, "open", t, None, canonical(inv)),
                    )
                )

                for j, h in enumerate(inv["hypotheses"]):
                    hid = f"{inv_id}_h{j}"
                    db_statements.append(
                        (
                            "INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?)",
                            (hid, inv_id, h, max(0.2, 0.8 - 0.2 * j), wall_ts),
                        )
                    )

                self.add_audit(
                    "investigation_opened",
                    {"investigation_id": inv_id, "entity": rid},
                    db_statements,
                )

        # Risk
        self.risk_scores = {}
        self.last_risk_components = {}

        for rid in ROBOT_IDS:
            trk = track_by_entity.get(rid)
            if trk is None:
                self.risk_scores[rid] = 0.0
                self.last_risk_components[rid] = {}
                continue

            res = ai_results[rid]
            obs = obs_by_entity.get(rid)

            anomaly = clamp(res["score"])
            criticality = zone_critical(trk.x, trk.y)
            persistence = clamp(self.detection_streak[rid] / 8.0)
            confidence = obs["confidence"] if obs else trk.conf
            evidence = clamp(
                min(1.0, len(trk.history) / 60.0) * 0.45
                + confidence * 0.35
                + (0.2 if rid in self.open_investigations else 0.0)
            )
            speed = math.hypot(trk.vx, trk.vy)
            impact = clamp(0.55 * criticality + 0.45 * min(1.0, speed / 90.0))

            components = {
                "anomaly": float(anomaly),
                "criticality": float(criticality),
                "persistence": float(persistence),
                "confidence": float(confidence),
                "evidence": float(evidence),
                "impact": float(impact),
            }

            risk = 100.0 * clamp(
                0.38 * anomaly
                + 0.17 * criticality
                + 0.15 * persistence
                + 0.12 * confidence
                + 0.10 * evidence
                + 0.08 * impact
            )

            self.risk_scores[rid] = float(risk)
            self.last_risk_components[rid] = components

        # Verification before new policy decision
        if self.active_action is not None and self.frame_idx >= int(
            self.active_action.get("verification_due", 0)
        ):
            act = self.active_action
            rid = act["entity_id"]
            trk = track_by_entity.get(rid)
            truth = true_poses[rid]

            rng = np.random.default_rng(777 + self.frame_idx)
            ind_x = truth["x"] + float(rng.normal(0, 0.7))
            ind_y = truth["y"] + float(rng.normal(0, 0.7))

            if trk is not None:
                after_error = math.hypot(trk.x - ind_x, trk.y - ind_y)
            else:
                after_error = 999.0

            before_error = float(act.get("before_error", 0.0))
            success = bool(after_error < max(10.0, before_error * 0.60))
            ver_conf = clamp(0.92 - after_error / 120.0, 0.35, 0.97)
            latency_s = (self.frame_idx - int(act.get("frame_idx", self.frame_idx))) / float(self.fps)

            verification_id = f"ver_{self.frame_idx:06d}_{rid}"
            verification = {
                "verification_id": verification_id,
                "action_id": act["action_id"],
                "frame_id": frame_id,
                "entity_id": rid,
                "success": success,
                "before_error": before_error,
                "after_error": float(after_error),
                "confidence": float(ver_conf),
                "latency_s": float(latency_s),
                "ts": wall_ts,
            }

            self.verifications.append(verification)
            self.verification_count += 1
            if success:
                self.successful_actions += 1

            evidence_id = f"evi_{self.frame_idx:06d}_{rid}"
            evidence_payload = {
                "evidence_id": evidence_id,
                "frame_id": frame_id,
                "frame_hash": frame_hash,
                "entity_id": rid,
                "action": act,
                "verification": verification,
                "model_version": self.ai.version,
                "config_version": CONFIG_VERSION,
                "risk_components": self.last_risk_components.get(rid, {}),
                "policy_decision": self.policy_decision,
                "policy_reason": self.last_policy_reason,
                "provenance": "synthetic-workcell",
            }
            evidence_hash = sha256_of(evidence_payload)
            evidence_payload["hash"] = evidence_hash
            self.evidence.append(evidence_payload)

            db_statements.append(
                (
                    "INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        verification_id,
                        act["action_id"],
                        frame_id,
                        int(success),
                        before_error,
                        after_error,
                        ver_conf,
                        wall_ts,
                        canonical(verification),
                    ),
                )
            )

            db_statements.append(
                (
                    "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?)",
                    (
                        evidence_id,
                        frame_id,
                        act["action_id"],
                        verification_id,
                        evidence_hash,
                        canonical(evidence_payload),
                        wall_ts,
                    ),
                )
            )

            self.add_audit(
                "verification_complete",
                {
                    "verification_id": verification_id,
                    "success": success,
                    "after_error": after_error,
                },
                db_statements,
            )

            self.active_action = None

        # Policy
        entity = None
        max_risk = 0.0
        if self.risk_scores:
            entity = max(self.risk_scores, key=lambda k: self.risk_scores[k])
            max_risk = self.risk_scores[entity]

        if entity is None or max_risk <= 0:
            decision = "deny"
            reason = "No active risk"
        elif max_risk < cfg["policy_threshold"] * 0.7:
            decision = "deny"
            reason = "Risk below escalation threshold"
        elif max_risk < cfg["policy_threshold"]:
            decision = "escalate"
            reason = "Risk elevated but below allow threshold"
        else:
            conf_entity = 0.0
            if entity in obs_by_entity:
                conf_entity = obs_by_entity[entity]["confidence"]
            elif entity in track_by_entity:
                conf_entity = track_by_entity[entity].conf

            evidence_component = self.last_risk_components.get(entity, {}).get("evidence", 0.0)

            if self.active_action is not None:
                decision = "deny"
                reason = "Action already in verification"
            elif t < cfg["min_action_time"]:
                decision = "escalate"
                reason = f"Waiting for demo action window t >= {cfg['min_action_time']:.0f}s"
            elif conf_entity < cfg["min_conf"]:
                decision = "escalate"
                reason = "Observation confidence too low"
            elif evidence_component < 0.35:
                decision = "escalate"
                reason = "Insufficient evidence quality"
            elif not cfg["policy_allow"]:
                decision = "escalate"
                reason = "Autonomous allow disabled in policy"
            else:
                decision = "allow"
                reason = "Risk, confidence, and evidence thresholds met"

        self.policy_decision = decision
        self.last_policy_reason = reason

        # Action
        if decision == "allow" and entity is not None:
            action_id = f"act_{self.frame_idx:06d}_{entity}"
            before_error = float(ai_results[entity]["residual"])

            action = {
                "action_id": action_id,
                "frame_id": frame_id,
                "frame_idx": self.frame_idx,
                "entity_id": entity,
                "kind": "SIM_RELOCALIZE",
                "policy": decision,
                "risk": float(max_risk),
                "before_error": before_error,
                "verification_due": self.frame_idx + int(5 * self.fps),
                "ts": wall_ts,
                "simulation_only": True,
            }

            self.active_action = action
            self.action_log.append(action)
            self.action_count += 1

            if entity == "R2":
                self.drift_corrected = True
                bx, by, bh = self.bias[entity]
                self.bias[entity] = (bx * 0.2, by * 0.2, bh * 0.2)

            db_statements.append(
                (
                    "INSERT OR REPLACE INTO actions VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        action_id,
                        frame_id,
                        entity,
                        action["kind"],
                        decision,
                        max_risk,
                        before_error,
                        wall_ts,
                        canonical(action),
                    ),
                )
            )

            self.add_audit(
                "action_allowed",
                {
                    "action_id": action_id,
                    "entity": entity,
                    "risk": max_risk,
                    "simulation_only": True,
                },
                db_statements,
            )

        # Metrics
        pose_errors = []
        for trk in active_tracks:
            if trk.entity_id in true_poses:
                truth = true_poses[trk.entity_id]
                pose_errors.append(math.hypot(trk.x - truth["x"], trk.y - truth["y"]))

        pose_error_mean = float(np.mean(pose_errors)) if pose_errors else 0.0
        processing_ms = (time.perf_counter() - start) * 1000.0
        self.last_processing_ms = processing_ms

        metric = {
            "frame_id": frame_id,
            "t": t,
            "fps_target": self.fps,
            "processing_ms": float(processing_ms),
            "active_tracks": len(active_tracks),
            "id_switches": self.tracker.id_switches,
            "observations": len(observations),
            "missed_obs": missed_obs,
            "ai_detections": sum(1 for r in ai_results.values() if r["det"]),
            "baseline_detections": sum(1 for r in ai_results.values() if r["baseline"]),
            "max_risk": float(max_risk),
            "pose_error_mean": pose_error_mean,
            "policy": self.policy_decision,
            "action": "active" if self.active_action else "idle",
            "verification": "pending" if self.active_action else ("verified" if self.verifications else "none"),
        }

        self.metrics.append(metric)

        db_statements.append(
            (
                "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)",
                (
                    f"met_{self.frame_idx:06d}",
                    frame_id,
                    wall_ts,
                    canonical(metric),
                ),
            )
        )

        self.last_frame_data = {
            "frame_id": frame_id,
            "hash": frame_hash,
            "t": t,
            "phase": demo_phase(t),
            "observations": observations,
            "detections": detections,
            "tracks": active_tracks,
            "ai": ai_results,
            "risks": self.risk_scores,
            "components": self.last_risk_components,
            "policy": self.policy_decision,
            "policy_reason": self.last_policy_reason,
            "true_poses": true_poses,
            "occluder": occluder,
            "active_action": self.active_action,
        }

        db_write_many(db_statements)

        if len(self.metrics) > 5000:
            self.metrics = self.metrics[-5000:]
        if len(self.eval_records) > 20000:
            self.eval_records = self.eval_records[-20000:]


# ======================================================================================================================
# CONFIG
# ======================================================================================================================
def current_config() -> Dict[str, Any]:
    return {
        "fps": int(st.session_state.get("fps", 6)),
        "noise": float(st.session_state.get("noise", 1.6)),
        "anomaly_strength": float(st.session_state.get("anomaly_strength", 1.0)),
        "scenario": st.session_state.get("scenario", "demo"),
        "policy_threshold": float(st.session_state.get("policy_threshold", 62.0)),
        "min_conf": float(st.session_state.get("min_conf", 0.55)),
        "min_action_time": float(st.session_state.get("min_action_time", 100.0)),
        "db_enabled": bool(st.session_state.get("db_enabled", True)),
        "policy_allow": bool(st.session_state.get("policy_allow", True)),
    }


# ======================================================================================================================
# RENDERING
# ======================================================================================================================
def render_frame(sim: Simulator, show_truth: bool = False, show_trajectories: bool = True) -> np.ndarray:
    data = sim.last_frame_data
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (32, 28, 24)

    # grid
    for x in range(0, WIDTH, 60):
        cv2.line(img, (x, 0), (x, HEIGHT), (44, 40, 36), 1)
    for y in range(0, HEIGHT, 60):
        cv2.line(img, (0, y), (WIDTH, y), (44, 40, 36), 1)

    # zones
    overlay = img.copy()
    for z in ZONES:
        color = (70, 170, 120) if z["critical"] < 0.8 else (70, 100, 220)
        cv2.rectangle(overlay, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), color, -1)
    img = cv2.addWeighted(overlay, 0.20, img, 0.80, 0)

    for z in ZONES:
        cv2.rectangle(img, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), (200, 180, 120), 1)
        cv2.putText(
            img,
            f"{z['name']} crit={z['critical']:.1f}",
            (z["x"] + 6, z["y"] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # occluder
    occ = data.get("occluder")
    if occ is not None:
        cv2.rectangle(img, (occ[0], occ[1]), (occ[0] + occ[2], occ[1] + occ[3]), (80, 80, 80), -1)
        cv2.rectangle(img, (occ[0], occ[1]), (occ[0] + occ[2], occ[1] + occ[3]), (160, 160, 160), 2)
        cv2.putText(img, "OCCLUDER", (occ[0] + 15, occ[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # trajectories
    if show_trajectories:
        for trk in data.get("tracks", []):
            color = ROBOT_MAP.get(trk.entity_id, {"color": (200, 200, 200)})["color"]
            pts = [(int(x), int(y)) for _, x, y in trk.history]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    # observations
    for obs in data.get("observations", []):
        color = ROBOT_MAP.get(obs["entity_id"], {"color": (200, 200, 200)})["color"]
        cx, cy = int(obs["x"]), int(obs["y"])
        cv2.circle(img, (cx, cy), 14, color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{obs['entity_id']} {obs['confidence']:.2f}",
            (cx + 16, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ground truth optional
    if show_truth:
        for rid, p in data.get("true_poses", {}).items():
            cv2.drawMarker(img, (int(p["x"]), int(p["y"])), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

    # active action marker
    act = data.get("active_action")
    if act is not None:
        trk_by_entity = {t.entity_id: t for t in data.get("tracks", [])}
        trk = trk_by_entity.get(act["entity_id"])
        if trk is not None:
            cv2.circle(img, (int(trk.x), int(trk.y)), 24, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                "SIM ACTION",
                (int(trk.x) + 22, int(trk.y) + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # HUD
    t = data.get("t", 0.0)
    phase = data.get("phase", "")
    max_risk = max(data.get("risks", {}).values(), default=0.0)
    policy = data.get("policy", "deny")

    cv2.putText(img, f"t={t:06.1f}s frame={sim.frame_idx}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, phase, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"latency={sim.last_processing_ms:.0f}ms risk={max_risk:.0f} policy={policy}",
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return img


# ======================================================================================================================
# ANALYTICS
# ======================================================================================================================
def expected_calibration_error(scores: List[float], labels: List[int], bins: int = 8) -> float:
    if not scores:
        return 0.0
    df = pd.DataFrame({"score": scores, "label": labels})
    if df.empty:
        return 0.0
    df["bin"] = pd.cut(df["score"].clip(0, 1), bins=bins, include_lowest=True)
    ece = 0.0
    total = len(df)
    for _, g in df.groupby("bin"):
        if len(g) == 0:
            continue
        ece += abs(float(g["score"].mean()) - float(g["label"].mean())) * len(g) / max(1, total)
    return float(ece)


def compute_analytics(sim: Simulator) -> Dict[str, Any]:
    metrics_df = pd.DataFrame(sim.metrics)
    eval_df = pd.DataFrame(sim.eval_records)

    out: Dict[str, Any] = {
        "metrics_df": metrics_df,
        "eval_df": eval_df,
        "p50": 0.0,
        "p95": 0.0,
        "avg_latency": 0.0,
        "throughput": 0.0,
        "continuity": 0.0,
        "id_switch_rate": 0.0,
        "false_alert_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "baseline_precision": 0.0,
        "baseline_recall": 0.0,
        "baseline_f1": 0.0,
        "ece": 0.0,
        "action_success": 0.0,
        "verification_latency": 0.0,
        "cpu_est": 0.0,
        "mem_est": 0.0,
        "cm": np.zeros((2, 2), dtype=int),
        "baseline_cm": np.zeros((2, 2), dtype=int),
    }

    if metrics_df.empty:
        return out

    out["p50"] = float(metrics_df["processing_ms"].quantile(0.5))
    out["p95"] = float(metrics_df["processing_ms"].quantile(0.95))
    out["avg_latency"] = float(metrics_df["processing_ms"].mean())
    out["throughput"] = float(min(st.session_state.get("fps", 6), 1000.0 / max(1.0, out["p50"])))

    expected_obs = max(1, len(metrics_df) * len(ROBOT_IDS))
    out["continuity"] = clamp(sim.tracker.associated_obs / expected_obs)
    out["id_switch_rate"] = sim.tracker.id_switches / max(1, sim.tracker.total_updates)

    if not eval_df.empty:
        y_true = eval_df["label"].astype(int).tolist()
        y_pred = eval_df["ai_pred"].astype(int).tolist()
        y_base = eval_df["baseline_pred"].astype(int).tolist()

        try:
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            out["precision"], out["recall"], out["f1"] = float(p), float(r), float(f)
        except Exception:
            pass

        try:
            bp, br, bf, _ = precision_recall_fscore_support(y_true, y_base, average="binary", zero_division=0)
            out["baseline_precision"], out["baseline_recall"], out["baseline_f1"] = float(bp), float(br), float(bf)
        except Exception:
            pass

        try:
            out["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
        except Exception:
            pass

        try:
            out["baseline_cm"] = confusion_matrix(y_true, y_base, labels=[0, 1])
        except Exception:
            pass

        normal_count = int((eval_df["label"] == 0).sum())
        false_positives = int(((eval_df["label"] == 0) & (eval_df["ai_pred"] == 1)).sum())
        out["false_alert_rate"] = false_positives / max(1, normal_count)

        out["ece"] = expected_calibration_error(
            eval_df["ai_score"].astype(float).tolist(),
            eval_df["label"].astype(int).tolist(),
        )

    out["action_success"] = sim.successful_actions / max(1, sim.verification_count)
    if sim.verifications:
        out["verification_latency"] = float(np.mean([v.get("latency_s", 5.0) for v in sim.verifications]))
    else:
        out["verification_latency"] = 0.0

    out["cpu_est"] = clamp(out["p95"] / 50.0, 0.02, 0.98) * 100.0
    out["mem_est"] = 20.0 + len(metrics_df) * 0.05 + len(eval_df) * 0.01

    return out


# ======================================================================================================================
# SESSION INITIALIZATION
# ======================================================================================================================
DEFAULTS = {
    "fps": 6,
    "running": True,
    "auto_run": True,
    "loop_demo": True,
    "scenario": "demo",
    "noise": 1.6,
    "anomaly_strength": 1.0,
    "policy_threshold": 62.0,
    "min_conf": 0.55,
    "min_action_time": 100.0,
    "show_truth": False,
    "show_trajectories": True,
    "db_enabled": True,
    "policy_allow": True,
    "page": "Overview",
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

if "sim" not in st.session_state:
    st.session_state.sim = Simulator(fps=st.session_state.fps)

sim: Simulator = st.session_state.sim
sim.fps = max(2, int(st.session_state.fps))

# ======================================================================================================================
# SIDEBAR
# ======================================================================================================================
PAGES = [
    "Overview",
    "Live Vision",
    "Tracking",
    "AI Detection",
    "Investigation",
    "Autonomous Action",
    "Verification",
    "Evidence & Audit",
    "Analytics/KPIs",
    "Simulator",
    "Configuration",
]

with st.sidebar:
    st.title("🤖 Pose Drift Tracker")
    st.caption("Simulator-only | Local-first | No API keys")

    page = st.radio("Page", PAGES, key="page")

    st.markdown("---")
    st.checkbox("Auto demo", key="auto_run")

    if st.button("Pause" if st.session_state.running else "Start", use_container_width=True):
        st.session_state.running = not st.session_state.running

    if st.button("Reset & Replay", use_container_width=True):
        sim.reset()
        st.session_state.running = True

    st.markdown("---")
    st.caption(f"Sim time: {sim.time:.1f}s / {DEMO_DURATION_S}s")
    st.caption(f"Frame: {sim.frame_idx}")
    st.caption(f"Phase: {demo_phase(sim.time)}")
    st.progress(clamp(sim.time / DEMO_DURATION_S))

    if st.session_state.get("db_error"):
        st.warning(f"DB note: {st.session_state.db_error}")

# ======================================================================================================================
# HEADER
# ======================================================================================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# ======================================================================================================================
# PAGE RENDER FUNCTIONS
# ======================================================================================================================
def render_overview():
    data = sim.last_frame_data

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sim time", f"{data['t']:.1f}s")
    c2.metric("Active tracks", len(data.get("tracks", [])))
    c3.metric("Max risk", f"{max(data.get('risks', {}).values(), default=0.0):.1f}")
    c4.metric("Policy", data.get("policy", "deny"))
    c5.metric("Actions", len(sim.action_log))
    c6.metric("Verifications", len(sim.verifications))

    st.info(f"Demo phase: {data.get('phase', '')}")
    st.progress(clamp(sim.time / DEMO_DURATION_S))

    img = render_frame(sim, st.session_state.show_truth, st.session_state.show_trajectories)
    st.image(img, caption="Synthetic workcell live view", use_container_width=True)

    with st.expander("Risk components"):
        if sim.last_risk_components:
            for rid, comp in sim.last_risk_components.items():
                st.markdown(f"**{rid}** — risk {sim.risk_scores.get(rid, 0.0):.1f}")
                st.json(comp)
        else:
            st.write("No risk components yet.")


def render_live_vision():
    data = sim.last_frame_data
    img = render_frame(sim, st.session_state.show_truth, st.session_state.show_trajectories)
    st.image(img, caption=f"Frame {data.get('frame_id', '')}", use_container_width=True)

    c1, c2 = st.columns(2)
    c1.code(f"frame hash: {data.get('hash', '')}", language=None)
    c2.write(f"Observations: {len(data.get('observations', []))} | Detections: {len(data.get('detections', []))}")

    det_rows = []
    for det in data.get("detections", []):
        det_rows.append(
            {
                "detection_id": det["detection_id"],
                "entity": det["entity_id"],
                "confidence": round(det["confidence"], 3),
                "bbox": det["bbox"],
            }
        )
    st.dataframe(pd.DataFrame(det_rows), use_container_width=True)


def render_tracking():
    data = sim.last_frame_data
    img = render_frame(sim, st.session_state.show_truth, True)
    st.image(img, caption="Tracking overlay", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("ID switches", sim.tracker.id_switches)
    c2.metric("Associated observations", sim.tracker.associated_obs)
    c3.metric("Total track updates", sim.tracker.total_updates)

    rows = []
    for trk in data.get("tracks", []):
        rows.append(
            {
                "track": f"T{trk.track_id:03d}",
                "entity": trk.entity_id,
                "x": round(trk.x, 1),
                "y": round(trk.y, 1),
                "heading": round(trk.heading, 2),
                "vx": round(trk.vx, 1),
                "vy": round(trk.vy, 1),
                "conf": round(trk.conf, 2),
                "miss": trk.miss,
                "active": trk.active,
                "history": len(trk.history),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_ai_detection():
    data = sim.last_frame_data
    ai = data.get("ai", {})

    cols = st.columns(len(ROBOT_IDS))
    for i, rid in enumerate(ROBOT_IDS):
        res = ai.get(rid, {})
        with cols[i]:
            st.markdown(f"### {rid}")
            st.progress(clamp(res.get("score", 0.0)))
            st.write(f"Score: {res.get('score', 0.0):.3f}")
            st.write(f"Residual: {res.get('residual', 0.0):.1f}px")
            st.write(f"Baseline: {'yes' if res.get('baseline') else 'no'}")
            st.write(f"Detection: {'yes' if res.get('det') else 'no'}")
            for e in res.get("explanation", []):
                st.caption(f"- {e}")

    hist_df = pd.DataFrame(sim.ai.history)
    if not hist_df.empty:
        fig = px.line(hist_df, x="t", y="score", color="rid", title="AI anomaly score over time")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(hist_df, x="t", y="residual", color="rid", title="Pose residual over time")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("AI history is accumulating.")


def render_investigation():
    if not sim.investigations:
        st.info("No investigations opened yet. Persistent detections will create cases automatically.")
        return

    for inv in sim.investigations:
        with st.expander(f"{inv['investigation_id']} — {inv['entity_id']} — {inv['status']}"):
            st.json(inv)

    st.markdown("### Case timeline")
    timeline_rows = []
    for inv in sim.investigations:
        timeline_rows.append(
            {
                "investigation_id": inv["investigation_id"],
                "entity": inv["entity_id"],
                "opened_at": inv["opened_at"],
                "status": inv["status"],
                "hypotheses": " | ".join(inv["hypotheses"]),
            }
        )
    st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True)


def render_autonomous_action():
    st.metric("Current policy decision", sim.policy_decision)
    st.info(sim.last_policy_reason)

    st.checkbox("Allow simulator-only autonomous action", key="policy_allow")
    st.caption("Actions only modify the virtual workcell. No hardware, no external systems.")

    action_rows = []
    for act in sim.action_log:
        action_rows.append(
            {
                "action_id": act["action_id"],
                "entity": act["entity_id"],
                "kind": act["kind"],
                "policy": act["policy"],
                "risk": round(act["risk"], 1),
                "before_error": round(act["before_error"], 1),
                "frame_idx": act.get("frame_idx", ""),
                "simulation_only": act.get("simulation_only", True),
            }
        )
    st.dataframe(pd.DataFrame(action_rows), use_container_width=True)


def render_verification():
    if not sim.verifications:
        st.info("No verifications yet. After an allowed action, an independent synthetic estimator verifies the result.")
        return

    rows = []
    for v in sim.verifications:
        rows.append(
            {
                "verification_id": v["verification_id"],
                "action_id": v["action_id"],
                "entity": v.get("entity_id", ""),
                "success": v["success"],
                "before_error": round(v["before_error"], 2),
                "after_error": round(v["after_error"], 2),
                "confidence": round(v["confidence"], 3),
                "latency_s": round(v.get("latency_s", 0.0), 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_evidence_audit():
    st.markdown("### Evidence")
    if sim.evidence:
        ev = sim.evidence[-1]
        st.code(f"Latest evidence SHA-256: {ev.get('hash', '')}", language=None)
        with st.expander("Latest evidence payload"):
            st.json(ev)

        rows = []
        for e in sim.evidence:
            rows.append(
                {
                    "evidence_id": e.get("evidence_id", ""),
                    "frame_id": e.get("frame_id", ""),
                    "action_id": e.get("action", {}).get("action_id", ""),
                    "hash": e.get("hash", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Evidence records are created after independent verification.")

    st.markdown("### Audit events")
    audit_rows = []
    for a in sim.audit[-200:]:
        audit_rows.append(
            {
                "event_id": a["event_id"],
                "kind": a["kind"],
                "payload": canonical(a["payload"]),
            }
        )
    st.dataframe(pd.DataFrame(audit_rows[::-1]), use_container_width=True)


def render_analytics():
    analytics = compute_analytics(sim)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("p50 latency (ms)", f"{analytics['p50']:.1f}")
    c2.metric("p95 latency (ms)", f"{analytics['p95']:.1f}")
    c3.metric("Throughput (fps)", f"{analytics['throughput']:.2f}")
    c4.metric("Track continuity", f"{analytics['continuity']:.2%}")
    c5.metric("ID-switch rate", f"{analytics['id_switch_rate']:.3%}")
    c6.metric("False alert rate", f"{analytics['false_alert_rate']:.3%}")

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("AI precision", f"{analytics['precision']:.3f}")
    c8.metric("AI recall", f"{analytics['recall']:.3f}")
    c9.metric("AI F1", f"{analytics['f1']:.3f}")
    c10.metric("Calibration error", f"{analytics['ece']:.3f}")
    c11.metric("Action success", f"{analytics['action_success']:.2%}")
    c12.metric("CPU est.", f"{analytics['cpu_est']:.1f}%")

    metrics_df = analytics["metrics_df"]
    if not metrics_df.empty:
        fig = px.line(metrics_df, x="t", y="pose_error_mean", title="Mean pose error")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(metrics_df, x="t", y="max_risk", title="Max risk")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(metrics_df, x="t", y="processing_ms", title="Processing latency (ms)")
        st.plotly_chart(fig, use_container_width=True)

    eval_df = analytics["eval_df"]
    if not eval_df.empty:
        st.markdown("### AI confusion matrix")
        fig = px.imshow(
            analytics["cm"],
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            text_auto=True,
            title="AI confusion matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Baseline confusion matrix")
        fig = px.imshow(
            analytics["baseline_cm"],
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            text_auto=True,
            title="Deterministic baseline confusion matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"Baseline P/R/F1 = {analytics['baseline_precision']:.3f} / "
            f"{analytics['baseline_recall']:.3f} / {analytics['baseline_f1']:.3f}"
        )


def render_simulator():
    st.subheader("Simulator controls")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start", use_container_width=True):
        st.session_state.running = True
    if c2.button("Pause", use_container_width=True):
        st.session_state.running = False
    if c3.button("Step", use_container_width=True):
        st.session_state.running = False
        sim.advance()
    if c4.button("Reset", use_container_width=True):
        sim.reset()
        st.session_state.running = True

    st.radio("Scenario", ["demo", "normal", "high"], key="scenario")
    st.checkbox("Loop demo", key="loop_demo")
    st.checkbox("Enable SQLite persistence", key="db_enabled")

    st.markdown("### Demo timeline")
    st.write(
        """
        - 0-20s: normal workcell  
        - 20-45s: unique anomaly injected into R2 coordinate frame  
        - 45-70s: persistent tracking through occlusion  
        - 70-90s: AI detection and explanation  
        - 90-105s: investigation and risk  
        - 105-115s: policy-gated simulator-only action  
        - 115-120s: independent verification, evidence hash, replay marker  
        """
    )

    st.metric("Current frame", sim.frame_idx)
    st.metric("Current time", f"{sim.time:.1f}s")
    st.metric("Phase", demo_phase(sim.time))


def render_configuration():
    st.subheader("Configuration")

    st.slider("FPS", 2, 15, value=int(st.session_state.fps), key="fps")
    st.slider("Sensor noise", 0.0, 5.0, 0.1, value=float(st.session_state.noise), key="noise")
    st.slider(
        "Anomaly strength",
        0.0,
        2.0,
        0.1,
        value=float(st.session_state.anomaly_strength),
        key="anomaly_strength",
    )
    st.slider(
        "Policy risk threshold",
        30.0,
        95.0,
        1.0,
        value=float(st.session_state.policy_threshold),
        key="policy_threshold",
    )
    st.slider(
        "Minimum confidence",
        0.1,
        0.95,
        0.01,
        value=float(st.session_state.min_conf),
        key="min_conf",
    )
    st.slider(
        "Minimum action time (s)",
        0.0,
        115.0,
        1.0,
        value=float(st.session_state.min_action_time),
        key="min_action_time",
    )

    st.checkbox("Show ground truth markers", key="show_truth")
    st.checkbox("Show trajectories", key="show_trajectories")
    st.checkbox("Allow simulator-only autonomous action", key="policy_allow")

    st.caption(
        "Changing FPS during a run changes simulated time mapping. For clean demo replay, press Reset after changing FPS."
    )


# ======================================================================================================================
# PAGE DISPATCH
# ======================================================================================================================
if page == "Overview":
    render_overview()
elif page == "Live Vision":
    render_live_vision()
elif page == "Tracking":
    render_tracking()
elif page == "AI Detection":
    render_ai_detection()
elif page == "Investigation":
    render_investigation()
elif page == "Autonomous Action":
    render_autonomous_action()
elif page == "Verification":
    render_verification()
elif page == "Evidence & Audit":
    render_evidence_audit()
elif page == "Analytics/KPIs":
    render_analytics()
elif page == "Simulator":
    render_simulator()
elif page == "Configuration":
    render_configuration()

# ======================================================================================================================
# AUTO ADVANCE LIVE DEMO
# ======================================================================================================================
if st.session_state.get("auto_run", True) and st.session_state.get("running", True):
    time.sleep(1.0 / max(2, int(st.session_state.get("fps", 6))))
    sim.advance()
    st.rerun()