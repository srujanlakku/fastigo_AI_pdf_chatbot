from typing import Dict, Iterable, List, Optional, Tuple

from langchain.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from .config import GOOGLE_API_KEY, MODEL_NAME, validate_api_settings
from .logger import logger
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class DocumentChatbot:
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = GOOGLE_API_KEY):
        validate_api_settings()
        self.model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key, streaming=True)

    def build_prompt(self, query: str, context: List[Dict[str, object]], conversation_context: Optional[object] = None) -> List[object]:
        context_text = "\n\n".join(
            [
                f"File: {item['metadata'].get('file_name', 'unknown')} | Page: {item['metadata'].get('page_number', 'unknown')}\n{item.get('document', '')}"
                for item in context
            ]
        )
        if isinstance(conversation_context, list):
            conversation_context = "\n".join(
                [f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_context if entry.get("content")]
            )

        prompt = USER_PROMPT_TEMPLATE.format(context=context_text, question=query)
        if conversation_context:
            prompt = f"Previous conversation:\n{conversation_context}\n\n" + prompt
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

    def format_sources(self, context: List[Dict[str, object]]) -> List[Dict[str, str]]:
        seen = {}
        for item in context:
            metadata = item.get("metadata", {})
            file_name = str(metadata.get("file_name", "unknown"))
            page_number = str(metadata.get("page_number", "unknown"))
            excerpt = str(item.get("document", "")).strip()
            key = f"{file_name}:{page_number}:{excerpt[:80]}"
            if key not in seen:
                seen[key] = {
                    "file_name": file_name,
                    "page_number": page_number,
                    "excerpt": excerpt[:400],
                    "relevance": round(float(item.get("fusion_score", item.get("semantic_score", 0.0)) or 0.0), 3),
                }
        return list(seen.values())

    def format_excerpts(self, context: List[Dict[str, object]]) -> List[str]:
        return [
            (
                f"{item.get('metadata', {}).get('file_name', 'unknown')} "
                f"(page {item.get('metadata', {}).get('page_number', 'unknown')}) — "
                f"{str(item.get('document', '')).strip()[:400]}"
            )
            for item in context
        ]

    @staticmethod
    def _extract_chunk_text(chunk: object) -> str:
        if isinstance(chunk, AIMessageChunk):
            content = chunk.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if hasattr(chunk, "text") and chunk.text:
            return str(chunk.text)
        if hasattr(chunk, "content") and chunk.content:
            return str(chunk.content)
        if hasattr(chunk, "message") and getattr(chunk.message, "content", None):
            return str(chunk.message.content)
        return ""

    def generate_answer(
        self,
        query: str,
        context: List[Dict[str, object]],
        conversation_history: Optional[object] = None,
    ) -> Tuple[str, List[Dict[str, str]], List[str]]:
        messages = self.build_prompt(query, context, conversation_context=conversation_history)
        try:
            response = self.model.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            sources = self.format_sources(context)
            excerpts = self.format_excerpts(context)
            return str(answer_text).strip(), sources, excerpts
        except Exception as exc:
            logger.error("Chat generation failed: %s", exc)
            raise RuntimeError("Unable to generate an answer. Please verify your Gemini API configuration.") from exc

    def stream_answer(
        self,
        query: str,
        context: List[Dict[str, object]],
        conversation_history: Optional[object] = None,
    ) -> Iterable[str]:
        messages = self.build_prompt(query, context, conversation_context=conversation_history)
        try:
            for chunk in self.model.stream(messages):
                token = self._extract_chunk_text(chunk)
                if token:
                    yield token
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            raise RuntimeError("Streaming response failed. Please verify your Gemini API configuration.") from exc
