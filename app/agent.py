from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

from .analyzer import (
    analyze_reference_video,
    analyze_target_video,
)


MODEL = "gemini-3.7-flash"


def analyze_reference_video_tool(
    video_path: str,
) -> dict:
    """
    Analyze a reference video.

    Use this tool whenever the user asks to analyze
    reference.mp4.

    The tool performs:

    - FFprobe metadata extraction
    - PySceneDetect scene detection
    - OpenCV representative frame extraction
    - OpenCV motion analysis
    - Gemini semantic video analysis
    - Pydantic validation

    Do not manually invent video-analysis results.
    """

    return analyze_reference_video(
        video_path
    )


def analyze_target_video_tool(
    target_video_path: str,
    reference_video_path: str,
) -> dict:
    """
    Analyze target.mp4.

    Uses:

    - FFprobe
    - PySceneDetect
    - OpenCV
    - Gemini

    Reference frames are supplied as style evidence.
    """

    return analyze_target_video(
        target_video_path,
        reference_video_path,
    )


root_agent = Agent(

    name="video_editing_agent",

    model=Gemini(
        model=MODEL,

        retry_options=types.HttpRetryOptions(
            attempts=3
        ),
    ),

    instruction="""
You are a reference-video and target-video analysis
agent for an AI video editing system.

When the user asks you to analyze a reference video:

1. Determine the video file path.

2. Call analyze_reference_video_tool.

3. Do not manually invent video-analysis results.

4. Return the tool's validated analysis.

The analysis must contain:

- segment IDs
- segment start times
- segment end times
- shot type
- camera motion
- subject motion
- editing characteristics
- transitions
- descriptions

When target.mp4 is being analyzed:

1. Call analyze_target_video_tool.

2. Do not manually invent target-analysis results.

3. Use the reference video as style evidence.

4. Return the validated target analysis.

You are an analysis agent.

Do not create an EditSpec yourself.
""",

    tools=[
        analyze_reference_video_tool,
        analyze_target_video_tool,
    ],
)