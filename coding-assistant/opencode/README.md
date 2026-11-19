# Build with OpenCode using Nscale Serverless Inference

Connect [OpenCode](https://opencode.ai) to an Nscale-hosted model, then use it to
implement and test a small Python feature. OpenCode reads files, proposes edits,
and runs commands on your computer; Nscale serves the model's inference requests.
You do not need a local GPU or a model deployment.

**Outcome:** an assistant that turns a written task into working code, with local
tests to check the result. Allow about 15–20 minutes, excluding account setup.

Use this Markdown guide for the terminal workflow, or open
[the companion notebook](walkthrough.ipynb) for executable setup and an API check.
The notebook hands off to a terminal for OpenCode's interactive interface.

## Before you start

- An Nscale account with inference credit and a service token. Follow the
  [Nscale quickstart](https://docs.nscale.com/docs/getting-started/quickstart).
- An exact model ID from **AI Services → Models** in the Nscale Console. Choose a
  text model that supports **streaming and tool calling**. Chat alone is not
  enough for an assistant that edits files and runs tests. Check the model's
  context and output limits as well.
- Python 3.10+, Git, Node.js with npm, and a terminal on macOS or Linux. The
  Python exercise and API check use only the standard library.
- OpenCode **1.19.3**. This recipe deliberately pins its configuration format;
  newer releases may require changes. See the [OpenCode provider
  guide](https://opencode.ai/docs/providers/) for the current format.

Inference calls consume your Nscale credit. The API check makes one request;
the coding task makes multiple requests, depending on the model's actions.
Review usage in the Nscale Console. No fixed price or token usage is assumed.

## 1. Install the tools and open the recipe

```bash
npm install -g opencode-ai@1.19.3
opencode --version
python3 --version
git clone https://github.com/nscaledev/nscale-cookbooks.git
cd nscale-cookbooks/coding-assistant/opencode
```

If you already cloned the repository, change to this recipe's directory instead.
The first version command should print `1.19.3`.

## 2. Set the connection details

Run these commands in the terminal you will use for OpenCode. Replace the model
placeholder with its exact API ID, including any organization prefix.

```bash
export NSCALE_BASE_URL="https://inference.api.nscale.com/v1"
export NSCALE_MODEL="REPLACE_WITH_MODEL_ID_FROM_NSCALE_CONSOLE"
printf 'Nscale service token (input hidden): '
read -r -s NSCALE_SERVICE_TOKEN
printf '\n'
export NSCALE_SERVICE_TOKEN
```

The token is entered without appearing in shell history. Keep it in your local
environment; do not paste it into prompts, files, screenshots, or notebook output.
The API uses Bearer authentication, as described in the
[Nscale chat guide](https://docs.nscale.com/docs/use-cases/chat).

## 3. Check the connection

From this recipe's directory, run:

```bash
python3 - <<'PY'
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

model = os.environ.get("NSCALE_MODEL", "")
token = os.environ.get("NSCALE_SERVICE_TOKEN", "")
if not model or model.startswith("REPLACE_") or not token:
    raise SystemExit("Set NSCALE_MODEL and NSCALE_SERVICE_TOKEN first.")
base_url = os.environ["NSCALE_BASE_URL"].rstrip("/")
request = Request(
    base_url + "/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "max_tokens": 4096,
        "stream": False,
    }).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
try:
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
except HTTPError as error:
    raise SystemExit(f"Nscale returned HTTP {error.code}; see Troubleshooting.") from None
except (URLError, TimeoutError):
    raise SystemExit("Could not reach Nscale; check your network and base URL.") from None
reply = result["choices"][0]["message"].get("content")
if not isinstance(reply, str) or not reply.strip():
    raise SystemExit("No text returned; check the model's output and reasoning settings.")
print(reply)
print("Usage:", result.get("usage", "not reported"))
PY
```

You should see a greeting and any reported token usage. This checks authentication,
the endpoint, and a text completion. The OpenCode exercise below separately checks
the streaming and tool-calling workflow.

## 4. Prepare an isolated practice project

The included `demo/` contains an unfinished function, its task description, agent
instructions, and four acceptance tests. Work in a temporary copy so you can
repeat the exercise without changing the cookbook.

```bash
practice_dir=$(mktemp -d "${TMPDIR:-/tmp}/nscale-opencode.XXXXXX")
cp -R demo/. "$practice_dir/"
cp opencode.json.example "$practice_dir/opencode.json"
cd "$practice_dir"
python3 -m unittest -v
```

**The initial test run must fail** with `NotImplementedError`. The assistant's
job is to implement `summarise_usage` according to `TASK.md`, preserving the tests.
The usage records are synthetic and require no access to your account data.

Review `opencode.json` before starting. It reads the token, endpoint, and model
from environment variables at runtime. It selects Nscale for both the main and
auxiliary model, and enables only that provider in this project configuration.
File edits and commands require your approval; file reads are allowed.
Your existing global OpenCode configuration can also affect behavior.

The example sets a **32,768-token context budget** and a **4,096-token output
budget**. These are recipe settings, not claims about your model's capacity.
Confirm both fit your model's limits, and adjust `limit.context`, `limit.output`,
and the API check's `max_tokens` if needed. Setting `tool_call: true` tells
OpenCode to use tools; it does not add tool support to a model.

The template uses the 1.19.3 fields `provider`, `npm`, and `options`. Their
versioned definitions are in OpenCode's
[provider configuration source](https://github.com/anomalyco/opencode/blob/v1.19.3/packages/core/src/v1/config/provider.ts).
It sends requests through the OpenAI-compatible Chat Completions adapter.

## 5. Ask OpenCode to implement the feature

In the practice directory, with the environment variables still set, start:

```bash
opencode --model "nscale/$NSCALE_MODEL"
```

Check that the selected provider and model match your Nscale configuration. Enter:

> Read AGENTS.md and TASK.md. Run the existing tests, implement summarise_usage
> in usage.py, and run the tests again. Keep the tests and configuration unchanged.
> Explain what changed and report the final test results.

Approve the file edits and test commands after reviewing them. You should see the
assistant read the task, observe failing tests, edit `usage.py`, and run the tests
successfully. Exact wording and tool sequences vary by model.

This is the key integration check: a streamed response that requests a local
tool, receives its result, and continues working. A model that only describes a
solution has not completed the exercise.

## 6. Verify the result yourself

Exit OpenCode with Ctrl+C, then run:

```bash
python3 -m unittest -v
cat usage.py
```

All four tests should pass. They check empty input, aggregation by model,
preserving input records, zero counts, and rejecting negative counts.
Compare `test_usage.py` and `TASK.md` with the originals in your clone to confirm
the assistant preserved them. Check your Nscale Console usage for the requests.

To repeat, start from a fresh temporary copy using step 4. When finished:

```bash
unset NSCALE_SERVICE_TOKEN
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| HTTP 401 or 403 | The service token is valid, unexpired, and authorized for inference. |
| HTTP 400 or model not found | Copy the exact model ID from the Console; check model-specific request parameters. |
| HTTP 404 | The base URL ends in `/v1`, rather than `/chat/completions`. |
| HTTP 429 | Check your credit, request limits, and the error details in the Console before retrying. |
| HTTP 5xx or timeout | Check service availability and retry later. |
| API check works, but OpenCode cannot connect | Launch OpenCode from the terminal containing the exported variables; check its version and project configuration. |
| Assistant outputs code but never edits files | Check tool-calling and streaming support for the chosen model; confirm you approved its tool requests. |
| Tool or reasoning format errors | Check the model's documented compatibility settings; do not assume every chat model has identical tool or reasoning behavior. |
| Context or output limit error | Adjust the configured budgets to fit the chosen model; start a fresh conversation if necessary. |

## Validation status

This recipe is a draft until a model has completed the live API check and
OpenCode exercise. A successful text completion alone does not establish coding
agent compatibility. Record the exact model ID and OpenCode version when
validating the recipe.

## References

- [Nscale quickstart](https://docs.nscale.com/docs/getting-started/quickstart)
- [Nscale Serverless models](https://docs.nscale.com/docs/ai-services/models)
- [Nscale Chat Completions API](https://docs.nscale.com/api-reference/inference/create-chat-completion)
- [OpenCode provider configuration](https://opencode.ai/docs/providers/)
- [OpenCode 1.19.3 configuration source](https://github.com/anomalyco/opencode/blob/v1.19.3/packages/core/src/v1/config/config.ts)
