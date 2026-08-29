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
# QWEN MODEL
# ============================================================

QWEN_MODEL = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen3-8B",
)


# ============================================================
# QWEN EDIT PLANNER
# ============================================================

class QwenEditPlanner:

    def __init__(self):

        print(
            f"[QWEN] Initializing model: {QWEN_MODEL}"
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
            model=QWEN_MODEL,
            timeout=120,
        )

        print(
            "[QWEN] Inference client initialized."
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
            "\n[QWEN] Generating EditSpec..."
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
execute using FFmpeg.

You must use ONLY the information provided in the
REFERENCE ANALYSIS and TARGET ANALYSIS.

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

            "operations": [],

            "rationale": "..."
        }}
    ],

    "global_operations": []
}}

IMPORTANT:

The final answer MUST be a single valid JSON object.
"""

        # ====================================================
        # QWEN MESSAGES
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
        # CALL QWEN
        # ====================================================

        print(
            "[QWEN] Sending reference and target "
            "analysis to Qwen3-8B..."
        )

        response = self.client.chat_completion(

            messages=messages,

            max_tokens=5000,

            temperature=0.1,

            # ------------------------------------------------
            # Qwen3 can produce a reasoning section.
            #
            # We don't need reasoning text here because the
            # output is supposed to be machine-readable JSON.
            # ------------------------------------------------

            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )

        # ====================================================
        # EXTRACT MESSAGE
        # ====================================================

        if not response.choices:

            raise RuntimeError(
                "Qwen returned no choices."
            )

        message = response.choices[0].message

        text = message.content

        if not text:

            raise RuntimeError(
                "Qwen returned an empty response."
            )

        print(
            "\n[QWEN] Raw response received."
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
                "Qwen returned invalid JSON.\n\n"
                f"JSON error: {e}\n\n"
                f"Qwen response:\n{text}"
            ) from e

        # ====================================================
        # PYDANTIC VALIDATION
        # ====================================================

        print(
            "[QWEN] Validating EditSpec..."
        )

        try:

            validated = EditSpec.model_validate(
                parsed
            )

        except Exception as e:

            raise ValueError(
                "Qwen returned JSON, but it does not "
                "match the EditSpec schema.\n\n"
                f"Validation error:\n{e}\n\n"
                f"Qwen JSON:\n"
                f"{json.dumps(parsed, indent=2)}"
            ) from e

        print(
            "[QWEN] EditSpec generated successfully."
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
    # EXTRACT JSON FROM QWEN RESPONSE
    # ========================================================

    @staticmethod
    def _extract_json(text: str):

        if not text:

            raise ValueError(
                "Qwen returned an empty response."
            )

        text = text.strip()

        # ----------------------------------------------------
        # Remove Markdown code fences if Qwen ignores the
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
                "Qwen did not return a JSON object.\n\n"
                f"Response:\n{text}"
            )

        if end <= start:

            raise ValueError(
                "Qwen returned malformed JSON boundaries.\n\n"
                f"Response:\n{text}"
            )

        return text[
            start:end + 1
        ]