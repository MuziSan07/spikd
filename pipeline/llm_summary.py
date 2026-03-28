"""
pipeline/llm_summary.py - Fixed version with system prompt and strict JSON parsing
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

    def generate(self, brain_result: dict, ad_copy: str = "") -> dict:
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
            fps=fps, ad_copy=ad_copy.strip(), sim_mode=sim_mode,
        )
        raw    = self._call_ollama(prompt)
        return self._parse(raw, bool(ad_copy.strip()))

    def _fmt(self, sec):
        return f"{int(sec) // 60}:{int(sec) % 60:02d}"

    def _build_prompt(self, duration_sec, peak_sec, drop_sec,
                      avg_eng, max_eng, fps, ad_copy, sim_mode):
        has_ad   = bool(ad_copy)
        sim_note = "[SIMULATION MODE]\n" if sim_mode else ""

        prompt = f"""{sim_note}WAVEFORM DATA:
Duration: {self._fmt(duration_sec)} | Peak: {self._fmt(peak_sec)} | Drop: {self._fmt(drop_sec)} | Avg: {avg_eng:.3f} | Max: {max_eng:.3f} | FPS: {fps:.1f}

Return ONLY this JSON object with no other text:
{{
  "summary": "one sentence clinical summary",
  "observations": [
    "{self._fmt(peak_sec)} [ALERT]: describe peak engagement finding",
    "{self._fmt(drop_sec)} [WARN]: describe engagement drop finding",
    "0:00 [NOTE]: describe hook strength finding",
    "0:12 [OK]: describe sustained engagement finding"
  ],
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "alignment": {"clinical alignment paragraph" if has_ad else ""},
  "alignment_score": {"High or Medium or Low" if has_ad else "N/A"}
}}"""

        if has_ad:
            prompt += f"\n\nAD COPY:\n{ad_copy}\n\nAnalyse alignment of ad copy with waveform peak at {self._fmt(peak_sec)} and drop at {self._fmt(drop_sec)}. Set alignment_score to High, Medium, or Low."

        return prompt.strip()

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "system": "You are a JSON API. Respond ONLY with a valid JSON object. No markdown. No explanation. No text before or after the JSON. Start with { and end with }.",
            "options": {
                "temperature": 0.0,
                "num_predict": 1200,
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

        # Try direct parse
        data = self._try_json(cleaned)

        # Try extract JSON object from text
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
            return {
                "summary":         str(data.get("summary", "Analysis complete.")),
                "observations":    observations,
                "key_findings":    [str(f) for f in data.get("key_findings", [])],
                "alignment":       str(data.get("alignment", "")),
                "alignment_score": str(data.get("alignment_score", "N/A")),
            }

        logger.warning("JSON parse failed — raw fallback")
        return {
            "summary":         "Analysis complete.",
            "observations":    ["0:00 [NOTE]: LLM response could not be parsed. Check Ollama logs."],
            "key_findings":    [],
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
            "key_findings": ["Waveform data is valid.", "LLM requires Ollama running."],
            "alignment": "",
            "alignment_score": "N/A",
        })
