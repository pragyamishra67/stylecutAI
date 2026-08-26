from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

from app.analyzer import analyze_reference_video


MODEL = "gemini-2.7-flash"


root_agent = Agent(
    name="reference_video_analyzer",

    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(
            attempts=3
        ),
    ),

    description=(
        "Analyzes a reference video using FFprobe, "
        "PySceneDetect, OpenCV and Gemini."
    ),

    instruction="""
You are a reference-video analysis agent.

Your job is to analyze a reference video and produce
a structured reference_analysis.json.

When the user asks you to analyze a reference video:

1. Determine the video file path.

2. Call the analyze_reference_video tool.

3. Do not manually invent video-analysis results.

4. The analysis tool performs:

   - FFprobe metadata extraction
   - PySceneDetect scene detection
   - OpenCV frame extraction
   - OpenCV motion analysis
   - Gemini semantic video analysis
   - Pydantic validation

5. Return the final analysis result to the user.

The final result must describe:

- segment IDs
- segment start times
- segment end times
- shot type
- camera motion
- subject motion
- editing characteristics
- transitions
- descriptions
""",

    tools=[
        analyze_reference_video
    ],
)