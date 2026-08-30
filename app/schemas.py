from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# VIDEO METADATA
# ============================================================

class VideoMetadata(BaseModel):
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int

    codec: Optional[str] = None
    pixel_format: Optional[str] = None


# ============================================================
# REPRESENTATIVE FRAME
# ============================================================

class RepresentativeFrame(BaseModel):
    timestamp: float
    path: str


# ============================================================
# REFERENCE SEGMENT
# ============================================================

class ReferenceSegment(BaseModel):

    segment_id: str

    start_time: float
    end_time: float
    duration: float

    shot_type: str
    camera_motion: str
    subject_motion: str

    editing_characteristics: List[str] = Field(
        default_factory=list
    )

    transitions: List[str] = Field(
        default_factory=list
    )

    description: str

    motion_score: Optional[float] = None


# ============================================================
# REFERENCE ANALYSIS
# ============================================================

class ReferenceAnalysis(BaseModel):

    video_path: str

    metadata: VideoMetadata

    total_segments: int

    segments: List[ReferenceSegment]


# ============================================================
# TARGET SEGMENT
# ============================================================

class TargetSegment(BaseModel):

    segment_id: str

    start_time: float
    end_time: float
    duration: float

    shot_type: str
    camera_motion: str
    subject_motion: str

    description: str

    motion_score: Optional[float] = None


# ============================================================
# TARGET ANALYSIS
# ============================================================

class TargetAnalysis(BaseModel):

    video_path: str

    metadata: VideoMetadata

    total_segments: int

    segments: List[TargetSegment]


# ============================================================
# EDIT OPERATION
# ============================================================

class EditOperation(BaseModel):

    operation: str

    parameters: dict = Field(
        default_factory=dict
    )


# ============================================================
# EDIT SPEC SEGMENT
# ============================================================

class EditSegment(BaseModel):

    output_segment_id: str

    source_segment_id: str

    source_start: float
    source_end: float

    target_start: float
    target_end: float

    operations: List[EditOperation] = Field(
        default_factory=list
    )

    rationale: str


# ============================================================
# EDIT SPEC
# ============================================================

class EditSpec(BaseModel):

    version: str = "1.0"

    source_video: str

    reference_video: str

    output_video: str

    reference_style_summary: str

    segments: List[EditSegment]

    global_operations: List[EditOperation] = Field(
        default_factory=list
    )