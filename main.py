from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
from pathlib import Path
import hashlib
import json
import zipfile
import os
import re

app = FastAPI(title="Testing Unit Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Auth (simple token-based)
# -------------------------

class User(BaseModel):
    id: str
    name: str
    role: str  # "supervisor" or "tester"

class LoginRequest(BaseModel):
    name: str  # we only send the username from frontend now

class LoginResponse(BaseModel):
    access_token: str
    role: str
    user: User

TOKENS: Dict[str, User] = {}

security = HTTPBearer()

async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    token = creds.credentials
    user = TOKENS.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

def require_role(required_role: str):
    async def dep(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dep

# Preset accounts
PRESET_TESTERS = ["tester1", "tester2", "tester3"]
PRESET_SUPERVISORS = ["supervisor"]

@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    username = body.name.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Name is required")

    key = username.lower()

    if key in PRESET_SUPERVISORS:
        role = "supervisor"
    elif key in PRESET_TESTERS:
        role = "tester"
    else:
        raise HTTPException(status_code=401, detail="Unknown user")

    user = User(id=str(uuid4()), name=username, role=role)
    token = str(uuid4())
    TOKENS[token] = user
    return LoginResponse(access_token=token, role=role, user=user)

@app.get("/testers", response_model=List[str])
def list_testers(supervisor: User = Depends(require_role("supervisor"))):
    """
    Scheduler uses this to populate tester dropdown.
    Only supervisors may fetch tester list.
    """
    return PRESET_TESTERS

# -------------------------
# Static Test Steps
# -------------------------

class TestStep(BaseModel):
    id: int
    name: str
    order: int
    required: bool = True

STEPS: List[TestStep] = [
    TestStep(id=1, name="Connectivity Test", order=1),
    TestStep(id=2, name="Functionality Test", order=2),
    TestStep(id=3, name="EIRP Determination & Stability Calibration", order=3),
    TestStep(id=4, name="Pre-Vibration Physical Layer Test", order=4),
    TestStep(id=5, name="Vibration Test", order=5),
    TestStep(id=6, name="Post-Vibration Physical Layer Test", order=6),
    TestStep(id=7, name="Thermal Cycling", order=7),
    TestStep(id=8, name="Post-Thermal Cycling Physical Layer Test", order=8),
    TestStep(id=9, name="Burn-in Test", order=9),
    TestStep(id=10, name="EMI/EMC Test", order=10),
    TestStep(id=11, name="Post-EMI/EMC Physical Layer Test", order=11),
    TestStep(id=12, name="BGAN Network Emulator Test", order=12),
]

STEP_BY_ID: Dict[int, TestStep] = {s.id: s for s in STEPS}
STEP_IDS_ORDERED = [s.id for s in sorted(STEPS, key=lambda s: s.order)]

@app.get("/steps", response_model=List[TestStep])
def list_steps(user: User = Depends(get_current_user)):
    return STEPS

# -------------------------
# Data Models (in-memory)
# -------------------------

class Unit(BaseModel):
    id: str
    sku: Optional[str] = None
    rev: Optional[str] = None
    lot: Optional[str] = None
    status: str = "IN_PROGRESS"  # or COMPLETED
    current_step_id: Optional[int] = None

class Assignment(BaseModel):
    id: str
    unit_id: str
    step_id: int
    tester_id: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    status: str = "PENDING"  # PENDING/RUNNING/DONE
    prev_passed: bool = False

class Result(BaseModel):
    id: str
    unit_id: str
    step_id: int
    passed: bool
    metrics: Dict[str, Any]
    files: List[str] = []
    submitted_by: Optional[str] = None
    finished_at: datetime

class FileMeta(BaseModel):
    id: str
    unit_id: str
    step_id: int
    result_id: str
    orig_name: str
    stored_name: str
    stored_path: str
    sha256: str
    size: int

class FileMetaOut(BaseModel):
    id: str
    unit_id: str
    step_id: int
    result_id: str
    orig_name: str

# === NOTIFICATIONS: model & storage ===================

class Notification(BaseModel):
    id: str
    tester_id: str
    unit_id: str
    from_step_id: int
    to_step_id: int
    message: str
    created_at: datetime
    read: bool = False


# All notifications in a flat dict by id
NOTIFICATIONS: Dict[str, Notification] = {}

# Index: tester_id -> list of notification IDs (to keep order)
TESTER_NOTIF_INDEX: Dict[str, List[str]] = {}


UNITS: Dict[str, Unit] = {}
ASSIGNMENTS: Dict[str, Assignment] = {}
RESULTS: Dict[str, Result] = {}
FILES: Dict[str, FileMeta] = {}


# For idempotency: (unit_id, step_id) -> result_id
RESULT_BY_UNIT_STEP: Dict[Tuple[str, int], str] = {}

@app.get("/results/{result_id}/files", response_model=List[FileMetaOut])
def list_result_files(result_id: str, user: User = Depends(get_current_user)):
    res = RESULTS.get(result_id)
    if not res:
        raise HTTPException(status_code=404, detail="Result not found")

    out: List[FileMetaOut] = []
    for fid in res.files:
        fm = FILES.get(fid)
        if not fm:
            continue
        out.append(
            FileMetaOut(
                id=fm.id,
                unit_id=fm.unit_id,
                step_id=fm.step_id,
                result_id=fm.result_id,
                orig_name=fm.orig_name,
            )
        )
    return out

@app.get("/files/{file_id}")
def download_file(file_id: str, user: User = Depends(get_current_user)):
    fm = FILES.get(file_id)
    if not fm:
        raise HTTPException(status_code=404, detail="File not found")

    path = Path(fm.stored_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path, filename=fm.orig_name)

@app.delete("/files/{file_id}")
def delete_file(file_id: str, supervisor: User = Depends(require_role("supervisor"))):
    fm = FILES.get(file_id)
    if not fm:
        raise HTTPException(status_code=404, detail="File not found")

    res = RESULTS.get(fm.result_id)
    if res:
        res.files = [fid for fid in res.files if fid != file_id]
        RESULTS[fm.result_id] = res

    p = Path(fm.stored_path)
    try:
        p.unlink()
    except FileNotFoundError:
        pass

    del FILES[file_id]

    return {"ok": True}

# -------------------------
# Units Endpoints
# -------------------------

class CreateUnitRequest(BaseModel):
    unit_id: str
    sku: Optional[str] = None
    rev: Optional[str] = None
    lot: Optional[str] = None

class UnitSummary(BaseModel):
    unit_id: str
    status: str
    progress_percent: float
    passed_steps: int
    total_steps: int
    next_step_id: Optional[int]
    next_step_name: Optional[str]

class UnitDetails(BaseModel):
    unit: Unit
    assignments: List[Assignment]
    results: List[Result]

@app.post("/units", response_model=Unit)
def create_unit(
    body: CreateUnitRequest,
    supervisor: User = Depends(require_role("supervisor")),
):
    unit_id = body.unit_id.strip()
    if not unit_id:
        raise HTTPException(status_code=400, detail="unit_id required")
    if unit_id in UNITS:
        raise HTTPException(status_code=400, detail="Unit already exists")

    unit = Unit(
        id=unit_id,
        sku=body.sku,
        rev=body.rev,
        lot=body.lot,
        current_step_id=STEP_IDS_ORDERED[0] if STEP_IDS_ORDERED else None,
    )
    UNITS[unit_id] = unit

    for idx, step_id in enumerate(STEP_IDS_ORDERED):
        a = Assignment(
            id=str(uuid4()),
            unit_id=unit_id,
            step_id=step_id,
            prev_passed=(idx == 0),
        )
        ASSIGNMENTS[a.id] = a

    return unit

@app.delete("/units/{unit_id}")
def delete_unit(
    unit_id: str,
    supervisor: User = Depends(require_role("supervisor")),
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")

    to_del_a = [aid for aid, a in ASSIGNMENTS.items() if a.unit_id == unit_id]
    for aid in to_del_a:
        del ASSIGNMENTS[aid]

    to_del_r = [rid for rid, r in RESULTS.items() if r.unit_id == unit_id]
    for rid in to_del_r:
        del RESULTS[rid]

    to_del_f = [fid for fid, f in FILES.items() if f.unit_id == unit_id]
    for fid in to_del_f:
        del FILES[fid]

    del UNITS[unit_id]
    return {"ok": True}

@app.get("/units/summary", response_model=List[UnitSummary])
def get_units_summary(user: User = Depends(get_current_user)):
    summaries: List[UnitSummary] = []
    for unit_id, unit in UNITS.items():
        unit_assignments = [a for a in ASSIGNMENTS.values() if a.unit_id == unit_id]
        unit_assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)
        total_steps = len(unit_assignments)
        passed_steps = 0
        for a in unit_assignments:
            key = (unit_id, a.step_id)
            rid = RESULT_BY_UNIT_STEP.get(key)
            if rid:
                res = RESULTS[rid]
                if res.passed:
                    passed_steps += 1
        progress = (passed_steps / total_steps) * 100 if total_steps else 0.0

        next_step_id = None
        next_step_name = None
        for a in unit_assignments:
            key = (unit_id, a.step_id)
            rid = RESULT_BY_UNIT_STEP.get(key)
            if not rid:
                next_step_id = a.step_id
                next_step_name = STEP_BY_ID[a.step_id].name
                break

        if total_steps == 0:
            status = "EMPTY"
        elif all(
            (RESULT_BY_UNIT_STEP.get((unit_id, a.step_id)) is not None)
            for a in unit_assignments
        ):
            status = "COMPLETED"
        else:
            status = "IN_PROGRESS"

        summaries.append(
            UnitSummary(
                unit_id=unit_id,
                status=status,
                progress_percent=progress,
                passed_steps=passed_steps,
                total_steps=total_steps,
                next_step_id=next_step_id,
                next_step_name=next_step_name,
            )
        )
    return summaries

@app.get("/units/{unit_id}/details", response_model=UnitDetails)
def get_unit_details(unit_id: str, user: User = Depends(get_current_user)):
    unit = UNITS.get(unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")
    assignments = [a for a in ASSIGNMENTS.values() if a.unit_id == unit_id]
    assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)
    results = [r for r in RESULTS.values() if r.unit_id == unit_id]
    return UnitDetails(unit=unit, assignments=assignments, results=results)

# -------------------------
# Tester Queue & Upcoming
# -------------------------

class TesterTask(BaseModel):
    assignment: Assignment
    step: TestStep
    reasons_blocked: List[str] = []

class TesterQueueResponse(BaseModel):
    ready: List[TesterTask]
    blocked: List[TesterTask]

def environment_ok_stub(
    unit_id: str, step_id: int
) -> Tuple[bool, Optional[str]]:
    return True, None

def calibration_ok_stub(
    unit_id: str, step_id: int
) -> Tuple[bool, Optional[str]]:
    return True, None

@app.get("/tester/queue", response_model=TesterQueueResponse)
def get_tester_queue(
    tester_id: str,
    user: User = Depends(get_current_user),
):
    ready: List[TesterTask] = []
    blocked: List[TesterTask] = []

    my_assignments: List[Assignment] = []
    for a in ASSIGNMENTS.values():
        if not (a.tester_id == tester_id or a.tester_id is None):
            continue
        if a.status not in ("PENDING", "RUNNING"):
            continue
        key = (a.unit_id, a.step_id)
        if key in RESULT_BY_UNIT_STEP:
            continue
        my_assignments.append(a)

    for a in my_assignments:
        reasons: List[str] = []

        if not a.prev_passed:
            reasons.append("Previous step not passed")

        ok_env, env_reason = environment_ok_stub(a.unit_id, a.step_id)
        if not ok_env:
            reasons.append(env_reason or "Environment out-of-range")

        ok_cal, cal_reason = calibration_ok_stub(a.unit_id, a.step_id)
        if not ok_cal:
            reasons.append(cal_reason or "Calibration expired")

        step = STEP_BY_ID[a.step_id]
        task = TesterTask(assignment=a, step=step, reasons_blocked=reasons)

        if reasons:
            blocked.append(task)
        else:
            ready.append(task)

    return TesterQueueResponse(ready=ready, blocked=blocked)

@app.get("/tester/assignments", response_model=List[Assignment])
def get_tester_assignments(
    tester_id: str,
    user: User = Depends(get_current_user),
):
    return [
        a
        for a in ASSIGNMENTS.values()
        if a.tester_id == tester_id and a.status in ("PENDING", "RUNNING")
    ]

# === NOTIFICATIONS: endpoints ========================

@app.get("/tester/notifications", response_model=List[Notification])
def get_tester_notifications(
    tester_id: str,
    unread_only: bool = False,
    user: User = Depends(get_current_user),
):
    """
    Return notifications for a tester.
    - Tester can only see their own.
    - Supervisor can see for any tester.
    Sorted by created_at (newest first).
    """
    if user.role == "tester" and user.name != tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    ids = TESTER_NOTIF_INDEX.get(tester_id, [])
    notes = [NOTIFICATIONS[nid] for nid in ids if nid in NOTIFICATIONS]

    if unread_only:
        notes = [n for n in notes if not n.read]

    # newest first
    notes.sort(key=lambda n: n.created_at, reverse=True)
    return notes


@app.post("/tester/notifications/{notif_id}/read")
def mark_notification_read(
    notif_id: str,
    user: User = Depends(get_current_user),
):
    """
    Mark a single notification as read.
    - Tester can only mark their own notifications.
    - Supervisor can mark any.
    """
    notif = NOTIFICATIONS.get(notif_id)
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")

    if user.role == "tester" and user.name != notif.tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    notif.read = True
    NOTIFICATIONS[notif_id] = notif
    return {"ok": True}


# -------------------------
# Results & Uploads
# -------------------------

class ResultIn(BaseModel):
    unit_id: str
    step_id: int
    metrics: Optional[Dict[str, Any]] = None  # optional
    passed: bool  # tester must decide
    finished_at: Optional[datetime] = None    # <- NEW


class ResultOut(BaseModel):
    id: str
    unit_id: str
    step_id: int
    passed: bool
    metrics: Dict[str, Any]
    files: List[str]
    submitted_by: Optional[str]
    finished_at: datetime

def update_assignments_after_result(unit_id: str, step_id: int, passed: bool) -> None:
    for a in ASSIGNMENTS.values():
        if a.unit_id == unit_id and a.step_id == step_id:
            a.status = "DONE"
            ASSIGNMENTS[a.id] = a
            break

    if step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_step_id = STEP_IDS_ORDERED[idx + 1]
            for a in ASSIGNMENTS.values():
                if a.unit_id == unit_id and a.step_id == next_step_id:
                    a.prev_passed = passed
                    ASSIGNMENTS[a.id] = a
                    break

    unit_assignments = [a for a in ASSIGNMENTS.values() if a.unit_id == unit_id]
    if unit_assignments and all(a.status == "DONE" for a in unit_assignments):
        unit = UNITS[unit_id]
        unit.status = "COMPLETED"
        UNITS[unit_id] = unit

# === NOTIFICATIONS: helper ==========================

def add_ready_notification(unit_id: str, from_step_id: int, to_step_id: int) -> None:
    """
    When a step passes and the next step becomes 'unblocked', notify the tester
    assigned to the next step (if any).
    Avoid sending duplicate notifications for the same unit/next-step/tester.
    """
    step = STEP_BY_ID.get(to_step_id)
    if not step:
        return

    # Find assignment for the next step that has a tester
    for a in ASSIGNMENTS.values():
        if a.unit_id == unit_id and a.step_id == to_step_id and a.tester_id:
            tester_id = a.tester_id

            # Check if a similar notification already exists
            existing_ids = TESTER_NOTIF_INDEX.get(tester_id, [])
            for nid in existing_ids:
                n = NOTIFICATIONS.get(nid)
                if not n:
                    continue
                if (
                    n.unit_id == unit_id
                    and n.from_step_id == from_step_id
                    and n.to_step_id == to_step_id
                ):
                    # Already notified this tester about this transition
                    return

            msg = f"Unit {unit_id} is ready for {step.name} (previous step passed)."
            nid = str(uuid4())
            notif = Notification(
                id=nid,
                tester_id=tester_id,
                unit_id=unit_id,
                from_step_id=from_step_id,
                to_step_id=to_step_id,
                message=msg,
                created_at=datetime.utcnow(),
                read=False,
            )
            NOTIFICATIONS[nid] = notif
            TESTER_NOTIF_INDEX.setdefault(tester_id, []).append(nid)
            return


@app.post("/results", response_model=ResultOut)
def create_or_update_result(
    body: ResultIn,
    user: User = Depends(get_current_user),
):
    if body.step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")
    if body.unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")

    passed = body.passed
    metrics = body.metrics or {}

    key = (body.unit_id, body.step_id)

    # Use client-provided finished_at if given, otherwise "now"
    finished = body.finished_at or datetime.utcnow()

    if key in RESULT_BY_UNIT_STEP:
        # update existing result (idempotent)
        rid = RESULT_BY_UNIT_STEP[key]
        existing = RESULTS[rid]
        existing.metrics = metrics
        existing.passed = passed
        existing.finished_at = finished
        RESULTS[rid] = existing
        res = existing
    else:
        rid = str(uuid4())
        res = Result(
            id=rid,
            unit_id=body.unit_id,
            step_id=body.step_id,
            metrics=metrics,
            passed=passed,
            files=[],
            submitted_by=user.id,
            finished_at=finished,
        )
        RESULTS[rid] = res
        RESULT_BY_UNIT_STEP[key] = rid


    # NEW LOGIC: whoever submits the result becomes the tester for this step
    for a in ASSIGNMENTS.values():
        if a.unit_id == body.unit_id and a.step_id == body.step_id:
            a.tester_id = user.name   # e.g. "tester2"
            ASSIGNMENTS[a.id] = a
            break

    # Update assignment chaining + unit status
    update_assignments_after_result(body.unit_id, body.step_id, passed)

    # If this step passed and there is a next step, notify that tester
    if passed and body.step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(body.step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_step_id = STEP_IDS_ORDERED[idx + 1]
            add_ready_notification(body.unit_id, body.step_id, next_step_id)

    return ResultOut(**res.dict())

# Storage config
STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(exist_ok=True)

ALLOWED_EXT = {".zip", ".csv", ".pdf", ".png"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

def sha256_fileobj(fileobj) -> str:
    h = hashlib.sha256()
    while True:
        chunk = fileobj.read(8192)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()

@app.post("/uploads")
async def upload_evidence(
    unit_id: str = Form(...),
    step_id: int = Form(...),
    result_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")
    if step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")
    if result_id not in RESULTS:
        raise HTTPException(status_code=404, detail="Result not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    content = await file.read()
    size = len(content)
    if size > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large")

    sha = hashlib.sha256(content).hexdigest()
    for existing in FILES.values():
        if existing.sha256 == sha and existing.unit_id == unit_id and existing.step_id == step_id:
            res = RESULTS[result_id]
            if existing.id not in res.files:
                res.files.append(existing.id)
                RESULTS[result_id] = res
            return {"file_id": existing.id, "deduplicated": True}

    bucket = sha[:2]
    bucket_dir = STORAGE_ROOT / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:6]
    server_name = f"{unit_id}_{step_id}_{timestamp}_{unique_suffix}{ext}"
    stored_path = bucket_dir / server_name

    with open(stored_path, "wb") as f:
        f.write(content)

    fid = str(uuid4())
    meta = FileMeta(
        id=fid,
        unit_id=unit_id,
        step_id=step_id,
        result_id=result_id,
        orig_name=file.filename,
        stored_name=server_name,
        stored_path=str(stored_path),
        sha256=sha,
        size=size,
    )
    FILES[fid] = meta

    res = RESULTS[result_id]
    res.files.append(fid)
    RESULTS[result_id] = res

    return {"file_id": fid, "deduplicated": False}

# -------------------------
# Scheduling (Supervisor)
# -------------------------


class DuplicateRequest(BaseModel):
    source_unit_id: str
    new_unit_ids: List[str]
    day_shift: int = 1


@app.post("/schedule/duplicate")
def duplicate_schedule(
    body: DuplicateRequest,
    supervisor: User = Depends(require_role("supervisor"))
):
    src = body.source_unit_id

    if src not in UNITS:
        raise HTTPException(status_code=404, detail="Source unit not found")

    src_assignments = [
        a for a in ASSIGNMENTS.values() if a.unit_id == src
    ]
    src_assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

    created_units = []

    for new_unit in body.new_unit_ids:
        if new_unit in UNITS:
            raise HTTPException(status_code=400, detail=f"Unit {new_unit} already exists")

        UNITS[new_unit] = Unit(
            id=new_unit,
            sku=UNITS[src].sku,
            rev=UNITS[src].rev,
            lot=UNITS[src].lot,
            status="IN_PROGRESS",
            current_step_id=STEP_IDS_ORDERED[0],
        )

        for src_a in src_assignments:

            def shift(dt):
                if not dt:
                    return None
                return dt + timedelta(days=body.day_shift)

            new_a = Assignment(
                id=str(uuid4()),
                unit_id=new_unit,
                step_id=src_a.step_id,
                tester_id=src_a.tester_id,
                start_at=shift(src_a.start_at),
                end_at=shift(src_a.end_at),
                status="PENDING",
                prev_passed=(src_a.step_id == STEP_IDS_ORDERED[0])
            )

            ASSIGNMENTS[new_a.id] = new_a

        created_units.append(new_unit)

    return {"ok": True, "created_units": created_units}


class AssignmentPatch(BaseModel):
    tester_id: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    status: Optional[str] = None


def overlaps(
    a_start: Optional[datetime],
    a_end: Optional[datetime],
    b_start: Optional[datetime],
    b_end: Optional[datetime],
) -> bool:
    """
    Check if two date ranges overlap.
    We treat them as full-day intervals, so any intersection counts as overlap.
    """
    if not a_start or not a_end or not b_start or not b_end:
        return False
    # inclusive overlap: [a_start, a_end] vs [b_start, b_end]
    return (a_start <= b_end) and (b_start <= a_end)


@app.get("/assignments/schedule", response_model=List[Assignment])
def get_schedule(supervisor: User = Depends(require_role("supervisor"))):
    # Just return all assignments; frontend can compute conflicts or show Gantt
    return list(ASSIGNMENTS.values())


@app.patch("/assignments/{assign_id}", response_model=Assignment)
def patch_assignment(
    assign_id: str,
    body: AssignmentPatch,
    supervisor: User = Depends(require_role("supervisor")),
):
    a = ASSIGNMENTS.get(assign_id)
    if not a:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # New values (dates only, but still stored as datetime)
    new_tester = body.tester_id if body.tester_id is not None else a.tester_id
    new_start = body.start_at if body.start_at is not None else a.start_at
    new_end = body.end_at if body.end_at is not None else a.end_at

    # Basic sanity: end cannot be before start
    if new_start and new_end and new_end < new_start:
        raise HTTPException(
            status_code=400,
            detail="End date cannot be before start date",
        )

    # (ONLY) prevent the **same unit** from having overlapping tests.
    # Tester is allowed to have multiple tests on the same day.
    for other in ASSIGNMENTS.values():
        if other.id == assign_id:
            continue
        if other.unit_id != a.unit_id:
            continue

        if overlaps(new_start, new_end, other.start_at, other.end_at):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unit '{a.unit_id}' already has another test scheduled "
                    f"from {other.start_at} to {other.end_at}"
                ),
            )

    # Apply changes
    if body.tester_id is not None:
        a.tester_id = body.tester_id
    if body.start_at is not None:
        a.start_at = body.start_at
    if body.end_at is not None:
        a.end_at = body.end_at
    if body.status is not None:
        a.status = body.status

    ASSIGNMENTS[assign_id] = a
    return a


# -------------------------
# Evidence Export (ZIP)
# -------------------------

ZIP_ROOT = Path("zips")
ZIP_ROOT.mkdir(exist_ok=True)

def step_folder_name(unit_id: str, step_id: int) -> str:
    step = STEP_BY_ID.get(step_id)
    if step:
        order = step.order
        name = step.name
    else:
        order = step_id
        name = f"Step {step_id}"
    return f"{order}. {unit_id}#{name}"

def sanitize_step_folder(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = name.strip().strip(".")
    return name or "step"

@app.get("/reports/unit/{unit_id}/zip")
def export_unit_zip(
    unit_id: str,
    user: User = Depends(get_current_user),
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")

    unit_results = [r for r in RESULTS.values() if r.unit_id == unit_id]
    unit_files = [f for f in FILES.values() if f.unit_id == unit_id]

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_name = f"{unit_id}_logs_{timestamp}.zip"
    zip_path = ZIP_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        results_json = [r.dict() for r in unit_results]
        zf.writestr("results.json", json.dumps(results_json, default=str, indent=2))

        lines = []
        for f in unit_files:
            lines.append(
                f"{f.id} | {f.unit_id} | {f.step_id} | {f.orig_name} | {f.stored_name}"
            )
        zf.writestr("manifest.txt", "\n".join(lines))

        for f in unit_files:
            path = Path(f.stored_path)
            if not path.exists():
                continue

            step = STEP_BY_ID.get(f.step_id)
            if step:
                folder_name = sanitize_step_folder(f"{step.order}. {unit_id}#{step.name}")
            else:
                folder_name = f"step_{f.step_id}"

            arcname = os.path.join(folder_name, f.orig_name)
            zf.write(path, arcname=arcname)

    return FileResponse(
        path=zip_path,
        filename=zip_name,
        media_type="application/zip",
    )

@app.get("/reports/unit/{unit_id}/step/{step_id}/zip")
def export_step_zip(
    unit_id: str,
    step_id: int,
    user: User = Depends(get_current_user),
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")
    if step_id not in STEP_BY_ID:
        raise HTTPException(status_code=404, detail="Step not found")

    step_files = [
        f
        for f in FILES.values()
        if f.unit_id == unit_id and f.step_id == step_id
    ]
    if not step_files:
        raise HTTPException(status_code=404, detail="No files for this step")

    folder = step_folder_name(unit_id, step_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_name = f"{folder}_logs_{timestamp}.zip"
    zip_path = ZIP_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in step_files:
            path = Path(f.stored_path)
            if not path.exists():
                continue
            arcname = os.path.join(folder, f.orig_name)
            zf.write(path, arcname=arcname)

    return FileResponse(
        path=zip_path,
        filename=zip_name,
        media_type="application/zip",
    )

# -------------------------
# Root
# -------------------------

@app.get("/")
def root():
    return {"message": "Testing Unit Tracker API running"}
