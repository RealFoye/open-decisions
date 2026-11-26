"""Score the example ticket with a local model, in-process (no server).

    pip install -e ".[local]"
    python examples/quickstart.py Qwen/Qwen2.5-0.5B-Instruct
"""

import json
import sys
from pathlib import Path

from opendecisions import DecisionRequest, DecisionService
from opendecisions.backends.local import LocalBackend

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
service = DecisionService(LocalBackend.from_pretrained(model_id))
request = DecisionRequest.model_validate_json(Path(__file__).with_name("request.json").read_text())
response = service.decide(request)

for key, answer in response.answers.items():
    print(key, json.dumps(answer.model_dump(exclude={"legend"}), ensure_ascii=False))
print(response.usage)
