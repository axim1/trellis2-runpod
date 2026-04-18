import traceback
from typing import Any, Dict

import runpod

from runpod_inference import GenerationOptions, TrellisRunpodRuntime, load_input_image


RUNTIME = TrellisRunpodRuntime()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (job or {}).get("input", {}) or {}

    try:
        image = load_input_image(input_data)
        options = GenerationOptions.from_input(input_data)
        return RUNTIME.generate(image, options)
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
