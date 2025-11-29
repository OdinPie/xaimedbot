"""
FastAPI server that wraps the diagnostic logic from `rag_with_test2.py`
and exposes it as a chat-style API for the clinical web interface.
"""

from __future__ import annotations
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os
import json
import re
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_with_test2 as core


YES_RESPONSES = {"yes", "y", "yeah", "yep", "sure", "affirmative", "ok", "okay", "continue"}
NO_RESPONSES = {"no", "n", "nope", "nah", "stop", "cancel", "exit", "quit"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ConditionSummary(BaseModel):
    name: str
    confidence: float
    matched_symptoms: List[str]
    denied_symptoms: List[str]
    penalty: float


class CreateSessionResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    stage: str
    phase: int
    top_conditions: Optional[List[ConditionSummary]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    stage: str
    phase: int
    top_conditions: Optional[List[ConditionSummary]] = None
    final_summary: Optional[str] = None


class ConsultationSession:
    """Stateful wrapper that mirrors `enhanced_multi_turn_session_with_tests`."""

    def __init__(self) -> None:
        self.symptom_tracker = core.EnhancedSymptomTracker()
        self.stage = "collecting_symptoms"
        self.phase = 1
        self.question_count = 0
        self.max_questions = 13
        self.last_question: Optional[str] = None
        self.last_symptom_asked: Optional[str] = None
        self.last_justification: Optional[str] = None
        self.ranked_diseases: List = []
        self.matched_symptoms: Dict[str, List[str]] = {}
        self.phase1_complete = False
        self.awaiting_phase2_consent = False
        self.finished = False
        self.phase2_queue: Deque[Dict] = deque()
        self.phase2_state: Dict[str, Dict] = {}
        self.current_test: Optional[Dict] = None
        self.final_candidates: List[str] = []
        self.final_summary: Optional[str] = None

    def start_session(self) -> List[ChatMessage]:
        intro = [
            "🏥 Hello! I’m your AI clinical assistant. I’ll ask a few questions "
            "to understand your symptoms. This can help surface conditions you "
            "can later discuss with your healthcare provider.",
            "To start, can you describe the symptoms you’re experiencing?",
        ]
        return [ChatMessage(role="assistant", content=msg) for msg in intro]

    def process_message(self, user_message: str) -> ChatResponse:
        if self.finished:
            return self._respond(["This consultation is complete. Please start a new session if needed."])

        cleaned = user_message.strip()
        if not cleaned:
            return self._respond(["I didn’t catch that. Could you share a bit more detail?"])

        lowered = cleaned.lower()
        if lowered == "exit":
            self.finished = True
            return self._respond(
                [
                    "Thank you for chatting. Please remember to consult with a healthcare professional "
                    "for personalized medical advice."
                ],
                mark_complete=True,
            )

        if self.stage == "collecting_symptoms":
            return self._handle_phase1_message(cleaned)
        if self.stage == "awaiting_phase2_consent":
            return self._handle_phase2_consent(cleaned)
        if self.stage == "collecting_tests":
            return self._handle_phase2_message(cleaned)

        return self._respond(["I’m wrapping things up. Feel free to start a new consultation anytime."])

    # ------------------------------------------------------------------ #
    # Phase 1 helpers
    # ------------------------------------------------------------------ #
    def _handle_phase1_message(self, user_message: str) -> ChatResponse:
        rag_context = self._build_rag_context(user_message)
        new_symptoms: List[str] = []

        if self.last_question and self.last_symptom_asked:
            top_names = [d for d, _ in self.symptom_tracker.disease_scores.items()][:3]
            extracted = core.followup_symptom_chain.run(
                question=self.last_question,
                answer=user_message,
                symptoms=", ".join(self.symptom_tracker.get_all_positive_symptoms()),
                top_diseases=", ".join(top_names),
                rag_context=rag_context,
            )
            intent = self._classify_yes_no(user_message)
            if intent == "yes":
                self.symptom_tracker.add_positive_symptom(self.last_symptom_asked)
                new_symptoms = [self.last_symptom_asked]
            elif intent == "no":
                self.symptom_tracker.add_negative_symptom(self.last_symptom_asked)
                new_symptoms = []
            else:
                new_symptoms = self._parse_symptoms(extracted)
                for symptom in new_symptoms:
                    self.symptom_tracker.add_positive_symptom(symptom)

            self.symptom_tracker.add_to_conversation(
                self.last_question, user_message, self.last_justification
            )
        else:
            extracted = core.symptom_chain.run(user_query=user_message, rag_context=rag_context)
            new_symptoms = self._parse_symptoms(extracted, include_raw=extracted)
            for symptom in new_symptoms:
                self.symptom_tracker.add_positive_symptom(symptom)

        all_text = " ".join(new_symptoms) if new_symptoms else user_message
        self.ranked_diseases, self.matched_symptoms = core.enhanced_diagnose(
            all_text, core.df_grouped, core.embedding_model, self.symptom_tracker
        )

        self.question_count += 1
        assessment = self._format_condition_assessment()
        responses = [assessment] if assessment else []

        if (
            not self.phase1_complete
            and core.should_continue_diagnosis(self.symptom_tracker, self.question_count)
            and self.question_count < self.max_questions
        ):
            next_question_payload = self._generate_followup_question()
            if next_question_payload:
                responses.extend(next_question_payload)
                return self._respond(responses)

        # Phase 1 complete
        self.phase1_complete = True
        self.stage = "awaiting_phase2_consent"
        summary = self._format_phase1_summary()
        responses.append(summary)
        responses.append(
            "Would you like me to ask about any recent diagnostic tests to help confirm these conditions? (yes/no)"
        )
        return self._respond(responses)

    def _parse_symptoms(self, extracted: str, include_raw: Optional[str] = None) -> List[str]:
        normalized_output = core.normalize_chain.run(raw_symptoms=extracted)
        combined = f"{include_raw or ''},{normalized_output}".strip(",")
        candidates = [
            s.strip().lower()
            for s in re.split(r"[,\n]+", combined)
            if s.strip().lower() not in {"none", "no symptoms", ""}
        ]
        return core.normalize_symptoms(candidates)

    def _generate_followup_question(self) -> Optional[List[str]]:
        next_question, symptom, justification = core.generate_llm_question(
            self.symptom_tracker, core.df_grouped
        )
        if not next_question or not symptom or not justification:
            return None

        self.last_question = next_question
        self.last_symptom_asked = symptom
        self.last_justification = justification
        return [
            f"💡 {justification}",
            f"👨‍⚕️ Doctor: {next_question}",
        ]

    def _format_condition_assessment(self) -> Optional[str]:
        if not self.ranked_diseases:
            return None

        lines = ["📊 Current leading conditions:"]
        for idx, (disease, score) in enumerate(self.ranked_diseases[:3], start=1):
            match_list = self.matched_symptoms.get(disease, [])
            confidence = self.symptom_tracker.get_disease_confidence_details(
                disease, score, len(match_list)
            )["confidence"]
            matches = ", ".join(match_list) if match_list else "none recorded yet"
            lines.append(f"{idx}. {disease} — confidence {confidence*100:.1f}% (matched: {matches})")
        return "\n".join(lines)

    def _format_phase1_summary(self) -> str:
        if not self.ranked_diseases:
            return (
                "I need more information to identify likely conditions. "
                "We can still review any tests if you’d like."
            )

        lines = [
            "✅ Symptom intake complete. Based on what you’ve shared, the top conditions are:",
        ]
        for idx, (disease, score) in enumerate(self.ranked_diseases[:3], start=1):
            match_list = self.matched_symptoms.get(disease, [])
            confidence = self.symptom_tracker.get_disease_confidence_details(
                disease, score, len(match_list)
            )["confidence"]
            lines.append(f"{idx}. {disease} — confidence {confidence*100:.1f}%")
        lines.append(
            "Diagnostic tests can help confirm or rule these out. Let me know if you’d like to discuss them."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Phase 2 helpers
    # ------------------------------------------------------------------ #
    def _handle_phase2_consent(self, user_message: str) -> ChatResponse:
        intent = self._classify_yes_no(user_message)
        if intent == "yes":
            self.phase = 2
            self.stage = "collecting_tests"
            self._prepare_phase2_tests()
            responses = [
                "Great! I’ll ask about specific lab or imaging results that are most useful for these conditions."
            ]
            next_prompt = self._ask_next_test()
            if next_prompt:
                responses.extend(next_prompt)
                return self._respond(responses)
            responses.extend(self._complete_phase2())
            return self._respond(responses)
        if intent == "no":
            self.finished = True
            self.stage = "completed"
            summary = self._format_phase1_summary()
            closing = "Thanks for the information. Please review these findings with your clinician."
            return self._respond([summary, closing], mark_complete=True)

        return self._respond(
            ["Please reply with yes or no so I know whether to continue with diagnostic tests."]
        )

    def _prepare_phase2_tests(self) -> None:
        self.phase2_queue.clear()
        self.phase2_state.clear()
        self.final_candidates = []

        diseases_to_eval = [d for d, _ in self.ranked_diseases[:3]]
        for disease in diseases_to_eval:
            info = core.DISEASE_INFO.get(disease)
            if not info:
                self.final_candidates.append(disease)
                continue

            thresholds = info.get("threshold_values", [])
            valid = [
                t
                for t in thresholds
                if t.get("test_name") not in {"N/A", "General"} and t.get("test_name")
            ]
            if not valid:
                self.final_candidates.append(disease)
                continue

            self.phase2_state[disease] = {
                "tests_total": len(valid),
                "tests_negative": 0,
                "still_possible": True,
            }
            for threshold in valid:
                self.phase2_queue.append(
                    {
                        "disease": disease,
                        "test_name": threshold["test_name"],
                        "threshold": threshold["threshold"],
                        "test_type": self._infer_test_type(threshold["test_name"]),
                        "full_text": threshold.get("full_text", ""),
                    }
                )

    def _ask_next_test(self) -> Optional[List[str]]:
        while self.phase2_queue:
            candidate = self.phase2_queue.popleft()
            disease_state = self.phase2_state.get(candidate["disease"])
            if not disease_state or not disease_state["still_possible"]:
                continue

            try:
                qa_response = core.test_question_chain.run(
                    disease=candidate["disease"],
                    test_name=candidate["test_name"],
                    threshold=candidate["threshold"],
                    test_type=candidate["test_type"],
                )
            except Exception as exc:
                print(f"Failed to build test question: {exc}")
                continue

            question, explanation = self._parse_test_question_response(qa_response)
            if not question:
                continue

            candidate["question"] = question
            candidate["explanation"] = explanation
            self.current_test = candidate
            messages = []
            if explanation:
                messages.append(f"💡 {explanation}")
            messages.append(f"👨‍⚕️ Doctor: {question}")
            return messages

        return None

    def _handle_phase2_message(self, user_message: str) -> ChatResponse:
        if not self.current_test:
            next_prompt = self._ask_next_test()
            if next_prompt:
                return self._respond(next_prompt)
            responses = self._complete_phase2()
            return self._respond(responses, mark_complete=True)

        test = self.current_test
        self.current_test = None
        try:
            interpretation = core.test_interpretation_chain.run(
                disease=test["disease"],
                test_name=test["test_name"],
                threshold=test["threshold"],
                patient_response=user_message,
            )
        except Exception as exc:
            print(f"Failed to interpret test response: {exc}")
            messages = [
                "I couldn’t interpret that test result. Let’s move to the next one.",
            ]
            next_prompt = self._ask_next_test()
            if next_prompt:
                messages.extend(next_prompt)
            else:
                messages.extend(self._complete_phase2())
            return self._respond(messages)

        status, explanation = self._parse_test_interpretation(interpretation)
        responses = [f"💡 {explanation}"] if explanation else []
        disease_state = self.phase2_state.get(test["disease"])

        if status == "BELOW_THRESHOLD":
            if disease_state:
                disease_state["tests_negative"] += 1
                if disease_state["tests_negative"] >= disease_state["tests_total"]:
                    disease_state["still_possible"] = False
                    self.symptom_tracker.rule_out_disease(test["disease"], explanation or "")
            self.symptom_tracker.add_test_result(test["disease"], test["test_name"], "negative", explanation or "")
            responses.append(f"❌ This result makes {test['disease']} less likely.")
        elif status == "MEETS_THRESHOLD":
            self.symptom_tracker.add_test_result(test["disease"], test["test_name"], "positive", explanation or "")
            responses.append(f"✅ This result supports {test['disease']}.")
        elif status == "NOT_DONE":
            responses.append("ℹ️ No problem—let’s continue with other tests.")
        else:
            responses.append("❓ I’m not fully sure about that result, but we can move on.")

        next_prompt = self._ask_next_test()
        if next_prompt:
            responses.extend(next_prompt)
            return self._respond(responses)

        responses.extend(self._complete_phase2())
        return self._respond(responses, mark_complete=True)

    def _complete_phase2(self) -> List[str]:
        remaining = [
            disease
            for disease, state in self.phase2_state.items()
            if state.get("still_possible")
        ]
        for disease in remaining:
            if disease not in self.final_candidates:
                self.final_candidates.append(disease)

        if not self.final_candidates and self.ranked_diseases:
            self.final_candidates = [d for d, _ in self.ranked_diseases[:2]]

        disease_info_str = json.dumps(
            {d: core.DISEASE_INFO.get(d, {}) for d in self.final_candidates[:3]}, indent=2
        )
        test_results_summary = json.dumps(self.symptom_tracker.test_results, indent=2)
        ruled_out_summary = ", ".join(self.symptom_tracker.diseases_ruled_out_by_tests) or "None"

        final_text = core.final_diagnosis_chain.run(
            symptoms=", ".join(self.symptom_tracker.get_all_positive_symptoms()),
            initial_diseases=", ".join([d for d, _ in self.ranked_diseases[:3]]),
            test_results=test_results_summary,
            ruled_out_diseases=ruled_out_summary,
            disease_info=disease_info_str,
        )

        self.final_summary = final_text
        self.finished = True
        self.stage = "completed"
        return [
            "📋 FINAL ASSESSMENT",
            final_text,
            "⚠️ This is an AI-generated discussion aid and not a medical diagnosis."
            " Please contact a licensed clinician for personal medical advice.",
        ]

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _respond(self, messages: List[str], mark_complete: bool = False) -> ChatResponse:
        payload = [ChatMessage(role="assistant", content=msg) for msg in messages]
        top_conditions = self._collect_top_conditions()
        if mark_complete:
            self.finished = True
            self.stage = "completed"
        return ChatResponse(
            session_id="",
            messages=payload,
            stage=self.stage,
            phase=self.phase,
            top_conditions=top_conditions,
            final_summary=self.final_summary,
        )

    def _collect_top_conditions(self) -> Optional[List[ConditionSummary]]:
        if not self.ranked_diseases:
            return None
        summaries = []
        for disease, score in self.ranked_diseases[:3]:
            matched = self.matched_symptoms.get(disease, [])
            details = self.symptom_tracker.get_disease_confidence_details(
                disease, score, len(matched)
            )
            confidence = details["confidence"]
            penalty = details["negative_impact"]
            denied = [
                symptom
                for symptom in self.symptom_tracker.negative_symptoms
                if self.symptom_tracker._symptom_belongs_to_disease(symptom, disease)
            ]
            summaries.append(
                ConditionSummary(
                    name=disease,
                    confidence=round(confidence * 100, 1),
                    matched_symptoms=matched,
                    denied_symptoms=denied,
                    penalty=round(penalty, 2),
                )
            )
        return summaries

    def _build_rag_context(self, text: str) -> str:
        if not core.vectorstore:
            return "No additional context available"
        try:
            docs = core.vectorstore.similarity_search(text, k=2)
            return "\n".join(doc.page_content for doc in docs)
        except Exception:
            return "No additional context available"

    @staticmethod
    def _classify_yes_no(text: str) -> Optional[str]:
        lowered = text.strip().lower()
        if lowered in YES_RESPONSES:
            return "yes"
        if lowered in NO_RESPONSES:
            return "no"
        return None

    @staticmethod
    def _infer_test_type(test_name: str) -> str:
        lowered = test_name.lower()
        if any(keyword in lowered for keyword in ["x-ray", "ct", "mri", "ultrasound", "scan"]):
            return "imaging test"
        if any(keyword in lowered for keyword in ["spirometry", "ekg", "ecg", "procedure"]):
            return "diagnostic procedure"
        return "laboratory test"

    @staticmethod
    def _parse_test_question_response(text: str) -> (Optional[str], Optional[str]):
        question = None
        explanation = None
        for line in text.strip().split("\n"):
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
        return question, explanation

    @staticmethod
    def _parse_test_interpretation(text: str) -> (Optional[str], Optional[str]):
        status = None
        explanation = None
        for line in text.strip().split("\n"):
            if line.startswith("STATUS:"):
                status = line.replace("STATUS:", "").strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
        return status, explanation


class SessionManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, ConsultationSession] = {}

    def create(self) -> (str, ConsultationSession):
        session_id = uuid.uuid4().hex
        session = ConsultationSession()
        self.sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> ConsultationSession:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("Session not found")
        return session


session_manager = SessionManager()


def backend_ready() -> bool:
    return bool(getattr(core, "SYSTEM_READY", False))


app = FastAPI(title="Clinical Diagnostic Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/styles.css")
def serve_css():
    return FileResponse(
        os.path.join(BASE_DIR, "styles.css"),
        media_type="text/css"
    )

@app.get("/app.js")
def serve_js():
    return FileResponse(
        os.path.join(BASE_DIR, "app.js"),
        media_type="application/javascript"
    )


@app.post("/api/session", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id, session = session_manager.create()
    intro_messages = session.start_session()
    top_conditions = session._collect_top_conditions()
    return CreateSessionResponse(
        session_id=session_id,
        messages=intro_messages,
        stage=session.stage,
        phase=session.phase,
        top_conditions=top_conditions,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        session = session_manager.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = session.process_message(request.message)
    response.session_id = request.session_id
    return response


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ready")
def readiness() -> Dict[str, bool]:
    return {"ready": backend_ready()}


