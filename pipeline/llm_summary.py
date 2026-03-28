"""
pipeline/llm_summary.py

Sends brain waveform data + optional ad copy to Llama 3.2 (via Ollama)
and returns a structured analysis.

Two modes:
  1. Waveform-only summary  — if no ad copy is provided
  2. Alignment analysis     — if ad copy is provided, compares ad intent
                              against the measured attention waveform
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("spikd.llm")

TIMEOUT = 120  # seconds


class LLMSummary:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.host  = ollama_host.rstrip("/")
        self.model = model

    # ── Public entry point ─────────────────────────────────────────────────────
    def generate(self, brain_result: dict, ad_copy: str = "") -> dict:
        """
        Parameters
        ----------
        brain_result : dict
            Output from TribeProcessor.process()
        ad_copy : str
            Optional ad script / headline text to align against.

        Returns
        -------
        dict:
            summary          : str   — brain engagement narrative
            alignment        : str   — alignment analysis (empty if no ad copy)
            alignment_score  : str   — "High / Medium / Low / N/A"
            key_findings     : list  — bullet list of key findings
        """
        waveform             = brain_result.get("waveform", [])
        peak_sec             = brain_result.get("peak_engagement_sec", 0)
        drop_sec             = brain_result.get("drop_timestamp_sec", 0)
        duration_sec         = brain_result.get("duration_sec", 0)
        simulation_mode      = brain_result.get("simulation_mode", False)

        # Summarise waveform numerically for the prompt
        wf_array       = waveform if isinstance(waveform, list) else list(waveform)
        avg_engagement = round(sum(wf_array) / len(wf_array), 3) if wf_array else 0
        max_engagement = round(max(wf_array), 3) if wf_array else 0

        prompt = self._build_prompt(
            duration_sec       = duration_sec,
            peak_sec           = peak_sec,
            drop_sec           = drop_sec,
            avg_engagement     = avg_engagement,
            max_engagement     = max_engagement,
            ad_copy            = ad_copy.strip(),
            simulation_mode    = simulation_mode,
        )

        raw_response = self._call_ollama(prompt)
        parsed       = self._parse_response(raw_response, bool(ad_copy.strip()))
        return parsed

    # ── Prompt construction ────────────────────────────────────────────────────
    def _build_prompt(
        self,
        duration_sec: float,
        peak_sec: float,
        drop_sec: float,
        avg_engagement: float,
        max_engagement: float,
        ad_copy: str,
        simulation_mode: bool,
    ) -> str:

        sim_note = (
            "\n[NOTE: Data was generated in simulation mode — "
            "real TRIBE v2 weights were not loaded. "
            "This analysis demonstrates the pipeline only.]\n"
            if simulation_mode else ""
        )

        base = f"""{sim_note}
You are a neuroscience-informed analyst for a clinical attention diagnostic tool called SPIKD.
You have been given the following brain engagement data derived from TRIBE v2 cortical surface modelling,
specifically from the Dorsal Attention Network (DAN) and Primary Visual Cortex (V1) as mapped
using the Glasser HCP MMP1.0 atlas.

BRAIN ENGAGEMENT DATA:
- Video duration:      {duration_sec:.1f} seconds
- Peak engagement at:  {peak_sec:.1f} seconds
- Engagement drop at:  {drop_sec:.1f} seconds
- Average engagement:  {avg_engagement:.3f} (normalised 0–1)
- Maximum engagement:  {max_engagement:.3f} (normalised 0–1)

TASK:
Write a precise, clinical-style report with the following JSON structure.
Return ONLY valid JSON, no markdown, no code blocks, no preamble.

{{
  "summary": "3–4 sentence paragraph describing the attention trajectory, key engagement moments, and clinical interpretation of the waveform.",
  "key_findings": [
    "Finding 1",
    "Finding 2",
    "Finding 3"
  ],
  "alignment": "{self._alignment_placeholder(ad_copy)}",
  "alignment_score": "{self._score_placeholder(ad_copy)}"
}}
"""

        if ad_copy:
            base += f"""
AD COPY / HEADLINE:
\"\"\"{ad_copy}\"\"\"

ALIGNMENT TASK:
In the "alignment" field, analyse whether the ad copy's intended message, emotional hooks,
and call-to-action timing align with the measured attention waveform.
Specifically address:
  1. Does the ad copy's hook land before or after the peak engagement at {peak_sec:.1f}s?
  2. Does the engagement drop at {drop_sec:.1f}s suggest the viewer lost attention before
     the primary message was delivered?
  3. What specific changes to copy timing or structure would improve neural alignment?

Set "alignment_score" to exactly one of: "High", "Medium", or "Low".
"""
        else:
            base += """
No ad copy was provided. Set "alignment" to "" and "alignment_score" to "N/A".
"""

        return base.strip()

    def _alignment_placeholder(self, ad_copy: str) -> str:
        return "TO BE FILLED BY MODEL" if ad_copy else ""

    def _score_placeholder(self, ad_copy: str) -> str:
        return "TO BE FILLED BY MODEL" if ad_copy else "N/A"

    # ── Ollama API call ────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":   0.3,
                "num_predict":   1024,
                "top_p":         0.9,
            },
        }

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not reachable at %s", self.host)
            return self._fallback_response()
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds", TIMEOUT)
            return self._fallback_response()
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return self._fallback_response()

    # ── Response parsing ───────────────────────────────────────────────────────
    def _parse_response(self, raw: str, has_ad_copy: bool) -> dict:
        # Strip any accidental markdown fences
        cleaned = raw.strip()
        for fence in ["```json", "```"]:
            cleaned = cleaned.replace(fence, "")
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return {
                "summary":         str(data.get("summary", "")),
                "key_findings":    list(data.get("key_findings", [])),
                "alignment":       str(data.get("alignment", "")),
                "alignment_score": str(data.get("alignment_score", "N/A")),
            }
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON — using raw text.")
            return {
                "summary":         raw.strip() or "Analysis unavailable.",
                "key_findings":    [],
                "alignment":       "",
                "alignment_score": "N/A",
            }

    # ── Fallback (Ollama offline) ──────────────────────────────────────────────
    def _fallback_response(self) -> str:
        return json.dumps({
            "summary": (
                "LLM analysis unavailable. Ollama service was not reachable. "
                "Ensure Ollama is running (ollama serve) and Llama 3.2 is pulled "
                "(ollama pull llama3.2) before running the pipeline."
            ),
            "key_findings": [
                "Brain waveform extracted successfully.",
                "Attention peak and drop timestamps are recorded in the data section.",
                "LLM narrative requires Ollama to be running.",
            ],
            "alignment":       "",
            "alignment_score": "N/A",
        })
