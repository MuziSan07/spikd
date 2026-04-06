"""
pipeline/llm_summary.py
Supports: target_audience, 3 recommendations, alignment scoring
"""

import json
import re
import logging
import requests

logger = logging.getLogger("spikd.llm")
TIMEOUT = 180


class LLMSummary:
    def __init__(self, ollama_host="http://localhost:11434", model="llama3.2"):
        self.host  = ollama_host.rstrip("/")
        self.model = model

    def generate(self, brain_result: dict, ad_copy: str = "", target_audience: str = "") -> dict:
        wf           = brain_result.get("waveform", [])
        peak_sec     = brain_result.get("peak_engagement_sec", 0)
        drop_sec     = brain_result.get("drop_timestamp_sec", 0)
        duration_sec = brain_result.get("duration_sec", 0)
        fps          = brain_result.get("fps", 30.0)
        sim_mode     = brain_result.get("simulation_mode", False)
        avg_eng      = round(sum(wf) / len(wf), 3) if wf else 0
        max_eng      = round(max(wf), 3)            if wf else 0

        prompt = self._build_prompt(
            duration_sec=duration_sec, peak_sec=peak_sec,
            drop_sec=drop_sec, avg_eng=avg_eng, max_eng=max_eng,
            fps=fps, ad_copy=ad_copy.strip(),
            target_audience=target_audience.strip(),
            sim_mode=sim_mode,
        )
        raw = self._call_ollama(prompt)
        return self._parse(raw, bool(ad_copy.strip()))

    def _fmt(self, sec):
        return f"{int(sec) // 60}:{int(sec) % 60:02d}"

    def _build_prompt(self, duration_sec, peak_sec, drop_sec,
                      avg_eng, max_eng, fps, ad_copy, target_audience, sim_mode):
        has_ad       = bool(ad_copy)
        has_audience = bool(target_audience)
        sim_note     = "[SIMULATION MODE]\n" if sim_mode else ""
        audience_note = f"TARGET AUDIENCE: {target_audience}\n" if has_audience else ""

        prompt = f"""{sim_note}{audience_note}WAVEFORM DATA:
Duration: {self._fmt(duration_sec)} | Peak: {self._fmt(peak_sec)} | Drop: {self._fmt(drop_sec)} | Avg: {avg_eng:.3f} | Max: {max_eng:.3f} | FPS: {fps:.1f}

Return ONLY this JSON object with no other text:
{{
  "summary": "one sentence summary of the attention performance",
  "observations": [
    "{self._fmt(peak_sec)} [ALERT]: describe peak engagement finding",
    "{self._fmt(drop_sec)} [WARN]: describe engagement drop finding",
    "0:00 [NOTE]: describe hook strength finding",
    "0:12 [OK]: describe sustained engagement finding"
  ],
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "recommendations": [
    "Recommendation 1: specific actionable edit with timestamp",
    "Recommendation 2: specific actionable edit with timestamp",
    "Recommendation 3: specific actionable edit with timestamp"
  ],
  "alignment": {"alignment paragraph if ad copy provided, else empty string"},
  "alignment_score": {"High or Medium or Low if ad copy provided, else N/A"}
}}"""

        if has_ad:
            prompt += f"\n\nAD COPY:\n{ad_copy}\n\nAnalyse alignment of ad copy with waveform peak at {self._fmt(peak_sec)} and drop at {self._fmt(drop_sec)}."
            if has_audience:
                prompt += f" Consider the target audience: {target_audience}. Grade whether the ad copy would resonate with this audience based on the waveform data."
            prompt += " Set alignment_score to High, Medium, or Low."
        elif has_audience:
            prompt += f"\n\nConsider the target audience: {target_audience}. Tailor your recommendations specifically for what would engage this audience."

        prompt += "\n\nFor recommendations: give 3 specific, actionable edits (e.g. 'Move the logo reveal to 3s to align with peak engagement'). Be concrete and timestamp-specific."

        return prompt.strip()

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "system": "You are a JSON API for video advertising analysis. Respond ONLY with a valid JSON object. No markdown. No explanation. No text before or after the JSON. Start with { and end with }.",
            "options": {
                "temperature": 0.2,
                "num_predict": 1500,
            },
        }
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not reachable")
            return self._fallback()
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            return self._fallback()

    def _parse(self, raw: str, has_ad: bool) -> dict:
        cleaned = raw.strip()
        cleaned = re.sub(r"```json", "", cleaned)
        cleaned = re.sub(r"```",     "", cleaned)
        cleaned = cleaned.strip()

        data = self._try_json(cleaned)

        if data is None:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = self._try_json(match.group(0))

        if data is not None:
            observations = data.get("observations", [])
            if not isinstance(observations, list):
                observations = []
            observations = [str(o) for o in observations if o]
            if not observations and data.get("summary"):
                observations = [f"0:00 [NOTE]: {data['summary']}"]

            recommendations = data.get("recommendations", [])
            if not isinstance(recommendations, list):
                recommendations = []
            recommendations = [str(r) for r in recommendations if r]

            return {
                "summary":          str(data.get("summary", "Analysis complete.")),
                "observations":     observations,
                "key_findings":     [str(f) for f in data.get("key_findings", [])],
                "recommendations":  recommendations,
                "alignment":        str(data.get("alignment", "")),
                "alignment_score":  str(data.get("alignment_score", "N/A")),
            }

        logger.warning("JSON parse failed — raw fallback")
        return {
            "summary":         "Analysis complete.",
            "observations":    ["0:00 [NOTE]: LLM response could not be parsed. Check Ollama logs."],
            "key_findings":    [],
            "recommendations": [],
            "alignment":       "",
            "alignment_score": "N/A",
        }

    def _try_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    def _fallback(self) -> str:
        return json.dumps({
            "summary": "LLM service unavailable.",
            "observations": [
                "0:00 [WARN]: Ollama not reachable on port 11434.",
                "0:00 [NOTE]: Run: ollama serve",
            ],
            "key_findings":    ["Waveform data is valid.", "LLM requires Ollama running."],
            "recommendations": [],
            "alignment":       "",
            "alignment_score": "N/A",
        })
