from typing import Optional

from pydantic import BaseModel, Field


class Motion(BaseModel):
    camera: str = Field(
        description=(
            "Camera movement: static, pan, tilt, zoom_in, zoom_out, "
            "tracking, handheld, orbit, or unknown."
        )
    )

    subject: Optional[str] = Field(
        default=None,
        description="Description of important subject movement."
    )

    intensity: Optional[str] = Field(
        default=None,
        description=(
            "Motion intensity: none, subtle, moderate, strong, "
            "or unknown."
        )
    )


class Editing(BaseModel):
    speed: str = Field(
        description=(
            "Playback speed: normal, slow_motion, fast_motion, "
            "or unknown."
        )
    )

    effect: Optional[str] = Field(
        default=None,
        description=(
            "Visible editing effect such as zoom, blur, flash, "
            "fade, color change, or none."
        )
    )

    crop_or_reframe: Optional[str] = Field(
        default=None,
        description=(
            "Any visible crop, reframing, punch-in, or composition change."
        )
    )


class Segment(BaseModel):
    id: str = Field(
        description="Segment ID such as R1, R2, R3."
    )

    start: float = Field(
        description="Start timestamp in seconds."
    )

    end: float = Field(
        description="End timestamp in seconds."
    )

    shot: str = Field(
        description=(
            "Shot type such as wide shot, medium shot, "
            "close-up, extreme close-up, POV, or unknown."
        )
    )

    motion: Motion

    editing: Editing

    description: str = Field(
        description="Concise description of what happens in this segment."
    )


class Transition(BaseModel):
    from_segment: str = Field(
        description="ID of the segment before the transition."
    )

    to_segment: str = Field(
        description="ID of the segment after the transition."
    )

    timestamp: float = Field(
        description="Timestamp of the transition in seconds."
    )

    type: str = Field(
        description=(
            "Transition type such as hard_cut, crossfade, "
            "fade_in, fade_out, wipe, dip_to_black, or unknown."
        )
    )


class ReferenceAnalysis(BaseModel):
    video_duration: float = Field(
        description="Total video duration in seconds."
    )

    segments: list[Segment] = Field(
        description="All detected segments in chronological order."
    )

    transitions: list[Transition] = Field(
        description="Transitions between consecutive segments."
    )