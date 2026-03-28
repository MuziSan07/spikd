"""
pipeline/llm_summary.py

Sends brain waveform data + optional ad copy to Llama 3.2 (via Ollama).
Returns structured clinical observations formatted as:

    0:04 [ALERT]: Text Overload detected. Activation spike at hook.
    0:11 [OK]: Sustained DAN engagement above 0.6 threshold.
    0:19 [WARN]: Attention decay rate exceeds -0.03/sec.
    0:27 [NOTE]: Hook landed before peak resonance at 0:31.

Two modes:
  1. Waveform-only  — no ad copy provided
  2. Alignment      — ad copy provided, scored High / Medium / Low
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("spikd.llm")

TIMEOUT = 120


class LLMSummary:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.host  = ollama_host.rstrip("/")
        self.model = model

    # ── Public entry point ─────────────────────────────────────────────────────
    def generate(self, brain_result: dict, ad_copy: str = "") -> dict:
        """
        Returns dict:
            summary          : str         — one-sentence summary
            observations     : list[str]   — clinical observation lines
            key_findings     : list[str]   — supporting bullet points
            alignment        : str         — alignment analysis (if ad copy given)
            alignment_score  : str         — High / Medium / Low / N/A
        """
        wf           = brain_result.get("waveform", [])
        peak_sec     = brain_result.get("peak_engagement_sec", 0)
        drop_sec     = brain_result.get("drop_timestamp_sec", 0)
        duration_sec = brain_result.get("duration_sec", 0)
        fps          = brain_result.get("fps", 30.0)
        sim_mode     = brain_result.get("simulation_mode", False)

        avg_eng = round(sum(wf) / len(wf), 3) if wf else 0
        max_eng = round(max(wf), 3)           if wf else 0

        prompt  = self._build_prompt(
            duration_sec=duration_sec,
            peak_sec=peak_sec,
            drop_sec=drop_sec,
            avg_eng=avg_eng,
            max_eng=max_eng,
            fps=fps,
            ad_copy=ad_copy.strip(),
            sim_mode=sim_mode,
        )

        raw    = self._call_ollama(prompt)
        parsed = self._parse(raw, bool(ad_copy.strip()))
        return parsed

    # ── Prompt ────────────────────────────────────────────────────────────────
    def _build_prompt(
        self,
        duration_sec, peak_sec, drop_sec,
        avg_eng, max_eng, fps,
        ad_copy, sim_mode,
    ) -> str:

        def fmt(sec):
            return f"{int(sec)//60}:{int(sec)%60:02d}"

        sim_note = (
            "[NOTE: SIMULATION MODE — synthetic data, not real TRIBE v2 output]\n"
            if sim_mode else ""
        )

        has_ad = bool(ad_copy)

        prompt = f"""{sim_note}
You are a clinical neuroscience analyst operating the SPIKD attention diagnostic system.
You have received TRIBE v2 cortical surface model output, atlas-masked to the
Dorsal Attention Network (DAN) and Primary Visual Cortex (V1) using Glasser MMP1.0.
The output has been collapsed to a 1D normalised attention waveform.

WAVEFORM METRICS:
  Video duration       : {fmt(duration_sec)} ({duration_sec:.1f}s)
  Peak resonance at    : {fmt(peak_sec)} ({peak_sec:.2f}s)
  Engagement drop at   : {fmt(drop_sec)} ({drop_sec:.2f}s)
  Average activation   : {avg_eng:.3f}  (normalised 0-1)
  Maximum activation   : {max_eng:.3f}  (normalised 0-1)
  Frame rate           : {fps:.1f} fps

CLINICAL OBSERVATION FORMAT:
Each observation must follow this exact format:
  M:SS [TAG]: Clinical finding in one concise sentence.

Tags must be exactly one of:
  [ALERT] — critical attention failure, overload, or sharp decay
  [OK]    — positive engagement signal, sustained activation
  [WARN]  — sub-optimal signal, moderate decay, or risk of dropout
  [NOTE]  — contextual finding, timing observation, or recommendation

Generate 4 to 6 observations distributed across the video timeline,
starting from 0:00 and progressing to the end of the video.
Each observation must reference a specific timestamp derived from the waveform data.

Return ONLY a valid JSON object. No markdown. No code blocks. No preamble.

{{
  "summary": "One sentence clinical summary of the attention profile.",
  "observations": [
    "0:00 [TAG]: Finding.",
    "0:XX [TAG]: Finding.",
    "0:XX [TAG]: Finding.",
    "0:XX [TAG]: Finding."
  ],
  "key_findings": [
    "Finding 1",
    "Finding 2",
    "Finding 3"
  ],
  "alignment": "{self._alignment_placeholder(has_ad)}",
  "alignment_score": "{self._score_placeholder(has_ad)}"
}}
"""

        if has_ad:
            prompt += f"""
AD COPY PROVIDED FOR ALIGNMENT ANALYSIS:
\"\"\"
{ad_copy}
\"\"\"

ALIGNMENT TASK:
In the "alignment" field, write a clinical paragraph (3-5 sentences) analysing
whether the ad copy's structure, hook timing, and call-to-action align with
the measured attention waveform. Specifically address:

  1. Does the ad copy's hook land before or after the peak resonance at {fmt(peak_sec)}?
  2. Does the engagement drop at {fmt(drop_sec)} suggest the viewer lost attention
     before the primary message or CTA was delivered?
  3. What specific structural change to the ad copy would improve neural alignment?

Set "alignment_score" to exactly one of: "High", "Medium", or "Low".
"""
        else:
            prompt += """
No ad copy provided.
Set "alignment" to "" and "alignment_score" to "N/A".
"""

        return prompt.strip()

    def _alignment_placeholder(self, has_ad: bool) -> str:
        return "TO BE FILLED BY MODEL" if has_ad else ""

    def _score_placeholder(self, has_ad: bool) -> str:
        return "TO BE FILLED BY MODEL" if has_ad else "N/A"

    # ── Ollama call ───────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.25,
                "num_predict": 1200,
                "top_p":       0.9,
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not reachable at %s", self.host)
            return self._fallback()
        except requests.exceptions.Timeout:
            logger.error("Ollama timed out after %ds", TIMEOUT)
            return self._fallback()
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            return self._fallback()

    # ── Parse response ────────────────────────────────────────────────────────
    def _parse(self, raw: str, has_ad: bool) -> dict:
        cleaned = raw.strip()
        for fence in ["```json", "```"]:
            cleaned = cleaned.replace(fence, "")
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return {
                "summary":         str(data.get("summary", "")),
                "observations":    list(data.get("observations", [])),
                "key_findings":    list(data.get("key_findings", [])),
                "alignment":       str(data.get("alignment", "")),
                "alignment_score": str(data.get("alignment_score", "N/A")),
            }
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON — using raw fallback.")
            return {
                "summary":         raw.strip() or "Analysis unavailable.",
                "observations":    [f"0:00 [NOTE]: {raw.strip()[:200]}"] if raw.strip() else [],
                "key_findings":    [],
                "alignment":       "",
                "alignment_score": "N/A",
            }

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _fallback(self) -> str:
        return json.dumps({
            "summary": (
                "LLM analysis unavailable. Ollama service not reachable. "
                "Ensure Ollama is running and Llama 3.2 is pulled."
            ),
            "observations": [
                "0:00 [WARN]: LLM service offline. Start Ollama: systemctl start ollama",
                "0:00 [NOTE]: Run 'ollama pull llama3.2' to download the model.",
                "0:00 [NOTE]: Waveform data is valid. Only narrative analysis is missing.",
            ],
            "key_findings": [
                "Brain waveform extracted successfully.",
                "Peak and drop timestamps are recorded in the stats table.",
                "LLM narrative requires Ollama to be running on port 11434.",
            ],
            "alignment":       "",
            "alignment_score": "N/A",
        })