import json
import os
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from .schemas import EditSpec


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# LLAMA INSTRUCT MODEL
# ============================================================

LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct",
)


# ============================================================
# LLAMA EDIT PLANNER
# ============================================================

class LlamaEditPlanner:

    def __init__(self):

        print(
            f"[LLAMA] Initializing model: {LLAMA_MODEL}"
        )

        # ----------------------------------------------------
        # Hugging Face Inference Client
        #
        # This uses the Hugging Face authentication that you
        # already configured with `hf auth login`.
        #
        # No model download is required.
        # ----------------------------------------------------

        self.client = InferenceClient(
            model=LLAMA_MODEL,
            timeout=120,
        )

        print(
            "[LLAMA] Inference client initialized."
        )


    # ========================================================
    # GENERATE EDIT SPEC
    # ========================================================

    def generate_editspec(
        self,
        reference_analysis,
        target_analysis,
    ):

        print(
            "\n[LLAMA] Generating EditSpec..."
        )

        # ----------------------------------------------------
        # Convert Pydantic models to dictionaries if needed
        # ----------------------------------------------------

        reference_data = self._to_dict(
            reference_analysis
        )

        target_data = self._to_dict(
            target_analysis
        )

        ref_meta = reference_data.get("metadata", {})
        reference_fps = float(ref_meta.get("fps", 30.0))
        reference_frame_count = int(ref_meta.get("frame_count", 0))
        reference_duration = float(
            ref_meta.get("duration")
            or ref_meta.get("duration_seconds")
            or (
                reference_frame_count / reference_fps
                if reference_fps
                else 0.0
            )
        )

        exact_reference_duration = (
            reference_frame_count / reference_fps
            if reference_fps > 0 and reference_frame_count > 0
            else reference_duration
        )


        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = f"""
You are an expert video editing planner.

Your job is to transform the target video so that its
editing structure and visual rhythm resemble the reference
video as closely as possible.

You are NOT editing the video yourself.

You are producing an EditSpec that another program will
execute using MoviePy.

You must use ONLY the information provided in the
REFERENCE ANALYSIS and TARGET ANALYSIS.

============================================================
REFERENCE TIMELINE CONSTRAINTS (HARD CONSTRAINTS)
============================================================

The reference video has:

FPS: {reference_fps}
Frame count: {reference_frame_count}
Duration: {reference_duration}
Exact Master Timeline Duration: {exact_reference_duration:.6f} seconds ({reference_frame_count} / {reference_fps})

The final output MUST use the same FPS as the reference ({reference_fps}).

The final output MUST contain EXACTLY the same number of frames
as the reference ({reference_frame_count} frames).

The final output duration MUST correspond exactly to:

reference_frame_count / reference_fps = {exact_reference_duration:.6f} seconds

Do not treat the duration as approximate.

Do not produce a final timeline with fewer or more frames.

Use the reference video as the master timeline.

PRIORITY ORDER:
1. EXACT reference frame count ({reference_frame_count})
2. EXACT reference FPS ({reference_fps})
3. EXACT reference timeline/duration ({exact_reference_duration:.6f} seconds)
4. Match reference cut/shot timing
5. Match reference actions as closely as possible
6. Match visual appearance/style

TIMING & ACTION MATCHING INSTRUCTIONS:
- Reproduce the reference video's cuts, actions, pacing, and timing to the maximum extent possible.
- Match number of shots, shot boundaries, cut positions, shot durations, pacing, action timing, movement timing, transitions, visual rhythm, and sequence of actions.
- When selecting target footage, choose the target segment that most closely corresponds to the action/content occurring at that reference position.
- If the target footage has different natural durations, use supported operations (trim, speed) to make the selected footage fit the exact reference timing.
- DO NOT invent footage or frames. Only select and edit footage that exists in the target video.
- If an exact action match is impossible, choose the closest available target footage while maintaining the exact reference timeline.

============================================================
REFERENCE ANALYSIS
============================================================

{json.dumps(
    reference_data,
    indent=2,
    ensure_ascii=False
)}

============================================================
TARGET ANALYSIS
============================================================

{json.dumps(
    target_data,
    indent=2,
    ensure_ascii=False
)}

============================================================
RULES
============================================================

1. Never invent target footage.

2. Every source_segment_id MUST correspond to an actual
   segment_id present in target_analysis.

3. Preserve the target video's actual visual content.

4. Match the reference video's:
   - pacing
   - shot duration
   - cut frequency
   - shot progression
   - camera-motion rhythm
   - transitions
   - montage structure

5. Select target footage that best corresponds to the
   reference segments.

6. If the reference has a 1.2 second shot, prefer a target
   shot with a similar useful duration.

7. Do not request effects that cannot be represented in the
   EditSpec schema.

8. Do not create footage that does not exist.

9. Do not use segment IDs that are not present in
   target_analysis.

10. source_start and source_end MUST lie within the selected
    target source segment.

11. target_start and target_end describe where the selected
    footage appears in the final output.

12. Keep the EditSpec internally consistent.

13. The final video MUST use the audio from
    data/reference.mp4, not the target video's audio.

14. Do not loop or extend the reference audio.

15. All segments MUST form one contiguous output timeline.
    The first target_start must be 0.0, and each segment's
    target_start must equal the previous segment's target_end.

16. The final target_end MUST correspond EXACTLY to:
    reference_frame_count / reference_fps = {exact_reference_duration:.6f} seconds.
    Do NOT treat the duration as approximate.

17. Each output segment MUST have the EXACT SAME DURATION
    as its corresponding reference segment.

18. The duration of a reference segment is a HARD constraint,
    not an approximation.

19. For every reference segment:

        reference_segment_duration =
        reference_end - reference_start

    The corresponding output segment MUST have exactly the
    same duration.

20. If the selected target footage is naturally shorter or
    longer than the required reference segment duration,
    use the existing supported operations, such as trim or
    speed, to make the output segment duration match
    the reference segment exactly.

21. The final output MUST preserve the same number of
    segments/shots as the reference wherever the available
    target footage allows it.

22. The start and end timing of every output segment MUST
    correspond to the start and end timing of its
    reference segment.

23. Do not allow cumulative timing drift between segments.
    Each segment must independently match its corresponding
    reference segment duration.

24. The reference video's frame timeline is the master
    timeline. Match the reference FPS and frame positions
    exactly.

25. If the reference contains more representative frames
    than the target analysis, this is NOT by itself an error.
    Representative-frame count does not need to match.
    What MUST match is the actual reference video timeline,
    including segment durations and final frame count.

26. When an exact action match is possible, select the target
    footage containing the same or most similar action.

27. Match the actions in the reference video to the maximum
    extent possible while still preserving the exact reference
    segment durations and timeline.

28. If the exact action is unavailable in the target video,
    choose the closest available target footage. Never invent
    an action or footage that does not exist.

29. The priority is:

        1. Exact reference FPS
        2. Exact reference frame count
        3. Exact reference segment durations
        4. Exact reference cut/timing positions
        5. Maximum possible action similarity
        6. Visual/style similarity

30. Before returning the EditSpec, verify that the duration
    of every output segment equals the duration of its
    corresponding reference segment.

============================================================
SUPPORTED OPERATIONS
============================================================

Only use these operations:

- trim
- speed
- scale
- crop
- fade_in
- fade_out
- volume
- mute

Do not create any other operation types.


============================================================
OPERATION PARAMETERS
============================================================

IMPORTANT:

Every operation MUST contain a "parameters" object.

The parameters object MUST contain the parameters required
by the operation.

Do NOT return an empty parameters object for an operation
that requires parameters.

Do NOT invent parameter names.

Use ONLY the parameter names specified below.


------------------------------------------------------------
1. trim
------------------------------------------------------------

Parameters:

- start: REQUIRED number
- end: OPTIONAL number

Example:

"operations": [
    {{
        "operation": "trim",
        "parameters": {{
            "start": 0.5,
            "end": 3.0
        }}
    }}
]

If "start" is not specified, use 0.0.

"end" should specify the ending time of the selected clip.


------------------------------------------------------------
2. speed
------------------------------------------------------------

Parameters:

- factor: REQUIRED number

The "factor" parameter is MANDATORY.

NEVER return:

"parameters": {{}}

for a speed operation.

The factor MUST be greater than 0.

Examples:

2.0 = 2x faster
1.0 = normal speed
0.5 = 2x slower

Example:

"operations": [
    {{
        "operation": "speed",
        "parameters": {{
            "factor": 2.0
        }}
    }}
]

Another example:

"operations": [
    {{
        "operation": "speed",
        "parameters": {{
            "factor": 0.5
        }}
    }}
]


------------------------------------------------------------
3. scale
------------------------------------------------------------

Parameters:

- width: OPTIONAL number
- height: OPTIONAL number

At least ONE of width or height MUST be provided.

Use the parameter names exactly as:

"width"
"height"

Example:

"operations": [
    {{
        "operation": "scale",
        "parameters": {{
            "width": 1080,
            "height": 1920
        }}
    }}
]

If only width is needed:

"operations": [
    {{
        "operation": "scale",
        "parameters": {{
            "width": 1080
        }}
    }}
]

If only height is needed:

"operations": [
    {{
        "operation": "scale",
        "parameters": {{
            "height": 1920
        }}
    }}
]


------------------------------------------------------------
4. crop
------------------------------------------------------------

Parameters:

- width: REQUIRED number
- height: REQUIRED number
- x: OPTIONAL number
- y: OPTIONAL number

The "width" and "height" parameters are MANDATORY.

"x" and "y" specify the crop position.

If x and y are not provided, the editor will center-crop
the video.

Example:

"operations": [
    {{
        "operation": "crop",
        "parameters": {{
            "width": 1080,
            "height": 1080,
            "x": 100,
            "y": 0
        }}
    }}
]

For a centered crop:

"operations": [
    {{
        "operation": "crop",
        "parameters": {{
            "width": 1080,
            "height": 1080
        }}
    }}
]


------------------------------------------------------------
5. fade_in
------------------------------------------------------------

Parameters:

- duration: OPTIONAL number

If duration is provided, it must be greater than or equal
to 0.

If duration is not provided, the editor uses its default.

Example:

"operations": [
    {{
        "operation": "fade_in",
        "parameters": {{
            "duration": 0.3
        }}
    }}
]


------------------------------------------------------------
6. fade_out
------------------------------------------------------------

Parameters:

- duration: OPTIONAL number

If duration is provided, it must be greater than or equal
to 0.

If duration is not provided, the editor uses its default.

Example:

"operations": [
    {{
        "operation": "fade_out",
        "parameters": {{
            "duration": 0.3
        }}
    }}
]


------------------------------------------------------------
7. volume
------------------------------------------------------------

Parameters:

- factor: OPTIONAL number
- volume: OPTIONAL number

Use "factor" as the preferred parameter name.

The value represents the volume multiplier.

Examples:

1.0 = normal volume
0.5 = half volume
0.0 = silent

Preferred example:

"operations": [
    {{
        "operation": "volume",
        "parameters": {{
            "factor": 0.5
        }}
    }}
]

Do NOT use both "factor" and "volume" in the same operation.

If volume is not required, do not generate a volume
operation.


------------------------------------------------------------
8. mute
------------------------------------------------------------

The mute operation does not require any parameters.

Use:

"operations": [
    {{
        "operation": "mute",
        "parameters": {{}}
    }}
]


============================================================
PARAMETER VALIDATION RULES
============================================================

Before returning the JSON, verify every operation.

For "trim":
- start must exist
- end may be omitted

For "speed":
- factor MUST exist
- factor MUST be greater than 0

For "scale":
- width and/or height must exist

For "crop":
- width MUST exist
- height MUST exist
- x and y are optional

For "fade_in":
- duration is optional

For "fade_out":
- duration is optional

For "volume":
- factor or volume may be used
- factor is preferred

For "mute":
- parameters must be an empty object

NEVER leave required parameters empty.

NEVER use parameter names other than the ones defined above.


============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY valid JSON.

Do not return:

- Markdown
- ```json
- explanations
- comments
- reasoning
- text before the JSON
- text after the JSON

The JSON must follow this exact structure:

{{
    "version": "1.0",

    "source_video": "data/target.mp4",

    "reference_video": "data/reference.mp4",

    "output_video": "outputs/edited.mp4",

    "reference_style_summary": "...",

    "segments": [
        {{
            "output_segment_id": "output_001",

            "source_segment_id": "segment_001",

            "source_start": 0.0,

            "source_end": 2.0,

            "target_start": 0.0,

            "target_end": 2.0,

            "operations": [
                {{
                    "operation": "speed",
                    "parameters": {{
                        "factor": 1.0
                    }}
                }}
            ],

            "rationale": "..."
        }}
    ],

    "global_operations": []
}}

IMPORTANT:

The final answer MUST be a single valid JSON object.

Every operation MUST contain a parameters object.

Required operation parameters MUST NOT be omitted.

For speed operations, "factor" MUST always be present.

For crop operations, "width" and "height" MUST always be present.

For scale operations, at least one of "width" or "height"
MUST be present.


============================================================
FINAL SELF-CHECK
============================================================

Before returning the JSON, check:

1. Is the response valid JSON?
2. Is there exactly one top-level JSON object?
3. Does every segment contain all required fields?
4. Does every operation contain "operation"?
5. Does every operation contain "parameters"?
6. Does every speed operation contain "factor"?
7. Is every speed factor greater than 0?
8. Does every crop operation contain "width" and "height"?
9. Does every scale operation contain "width" and/or "height"?
10. Are all source_segment_id values valid?
11. Are source_start/source_end within the selected segment?
12. Are target_start/target_end contiguous?
13. Does the final target_end equal EXACTLY {exact_reference_duration:.6f}
    (reference_frame_count / reference_fps)?
14. Is reference audio used instead of target audio?
15. Is reference audio neither looped nor extended?
16. Are all the edited video segments equal in duration to the segments of reference video?

Return ONLY the final JSON object.
"""


        # ====================================================
        # LLAMA MESSAGES
        # ====================================================

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video editing planner. "
                    "You produce strict JSON EditSpec objects. "
                    "Return ONLY valid JSON. "
                    "Do not include explanations or Markdown."
                ),
            },

            {
                "role": "user",
                "content": prompt,
            },

        ]


        # ====================================================
        # CALL LLAMA
        # ====================================================

        print(
            "[LLAMA] Sending reference and target "
            "analysis to Llama Instruct..."
        )

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=10000,
            temperature=0.1,
        )


        # ====================================================
        # EXTRACT MESSAGE
        # ====================================================

        if not response.choices:

            raise RuntimeError(
                "Llama returned no choices."
            )

        message = response.choices[0].message

        text = message.content

        if not text:

            raise RuntimeError(
                "Llama returned an empty response."
            )

        print(
            "\n[LLAMA] Raw response received."
        )


        # ====================================================
        # EXTRACT JSON
        # ====================================================

        json_text = self._extract_json(
            text
        )


        # ====================================================
        # PARSE JSON
        # ====================================================

        try:

            parsed = json.loads(
                json_text
            )

        except json.JSONDecodeError as e:

            raise ValueError(
                "Llama returned invalid JSON.\n\n"
                f"JSON error: {e}\n\n"
                f"Llama response:\n{text}"
            ) from e


        # ====================================================
        # PYDANTIC VALIDATION
        # ====================================================

        print(
            "[LLAMA] Validating EditSpec..."
        )

        try:

            validated = EditSpec.model_validate(
                parsed
            )

        except Exception as e:

            raise ValueError(
                "Llama returned JSON, but it does not "
                "match the EditSpec schema.\n\n"
                f"Validation error:\n{e}\n\n"
                f"Llama JSON:\n"
                f"{json.dumps(parsed, indent=2)}"
            ) from e


        print(
            "[LLAMA] EditSpec generated successfully."
        )

        return validated.model_dump()


    # ========================================================
    # CONVERT INPUT TO DICTIONARY
    # ========================================================

    @staticmethod
    def _to_dict(data):

        # ----------------------------------------------------
        # Pydantic v2 model
        # ----------------------------------------------------

        if isinstance(
            data,
            BaseModel
        ):

            return data.model_dump()


        # ----------------------------------------------------
        # Already a dictionary
        # ----------------------------------------------------

        if isinstance(
            data,
            dict
        ):

            return data


        # ----------------------------------------------------
        # Anything else
        # ----------------------------------------------------

        raise TypeError(
            "Analysis data must be either a dictionary "
            "or a Pydantic BaseModel."
        )


    # ========================================================
    # EXTRACT JSON FROM LLAMA RESPONSE
    # ========================================================

    @staticmethod
    def _extract_json(text: str):

        if not text:

            raise ValueError(
                "Llama returned an empty response."
            )

        text = text.strip()


        # ----------------------------------------------------
        # Remove Markdown code fences if Llama ignores the
        # instruction and returns ```json ... ```
        # ----------------------------------------------------

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()


        # ----------------------------------------------------
        # Find JSON object
        # ----------------------------------------------------

        start = text.find("{")

        end = text.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                "Llama did not return a JSON object.\n\n"
                f"Response:\n{text}"
            )

        if end <= start:

            raise ValueError(
                "Llama returned malformed JSON boundaries.\n\n"
                f"Response:\n{text}"
            )

        return text[
            start:end + 1
        ]

