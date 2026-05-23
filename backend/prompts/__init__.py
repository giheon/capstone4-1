"""
Prompt templates module (deprecated)

Note: 프롬프트는 이제 config.py에서 중앙 관리됩니다.
이 모듈은 하위 호환성을 위해 유지됩니다.
"""
# Re-export from config for backwards compatibility
import sys
sys.path.append('..')

try:
    from config import (
        OCR_ROUTING_SYSTEM_PROMPT,
        OCR_ROUTING_USER_PROMPT,
        EXPLANATION_SYSTEM_PROMPT,
        EXPLANATION_PROMPTS
    )

    __all__ = [
        "OCR_ROUTING_SYSTEM_PROMPT",
        "OCR_ROUTING_USER_PROMPT",
        "EXPLANATION_SYSTEM_PROMPT",
        "EXPLANATION_PROMPTS"
    ]
except ImportError:
    __all__ = []
