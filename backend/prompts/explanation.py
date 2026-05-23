"""
Prompt templates for math explanation generation.

New Architecture:
- 9 prompts (3 subjects × 3 levels)
- LaTeX 미사용
- JSON 출력 형식

Note: Main prompts are defined in graph/state.py (EXPLANATION_PROMPTS)
This module contains helper prompts and constants.
"""

# OCR + Routing 통합 프롬프트
OCR_ROUTING_PROMPT = """이미지에서 수학 문제를 분석하세요.

## 작업
1. 문제 텍스트를 정확하게 추출
2. 문제 유형 판별 (객관식/주관식)
3. 과목 분류
4. 난이도 분류
5. 단원 분류

## 과목별 단원
- 확률과통계: 경우의 수, 확률, 확률분포, 통계적 추정
- 미적분: 수열, 미분법, 적분법
- 기하: 이차곡선, 평면벡터, 공간도형과 공간좌표

## 출력 (JSON)
{
    "problem_text": "문제 전체 내용",
    "question_type": "objective 또는 subjective",
    "subject": "확률과통계/미적분/기하 중 하나",
    "difficulty": "쉬움/보통/어려움/킬러 중 하나",
    "unit": "해당 과목의 단원명"
}

## 문제 유형 판별 기준
- 보기에 ①②③④⑤가 있으면 → "objective"
- 숫자를 직접 답하는 형식이면 → "subjective"
"""


# 해설 생성 시스템 프롬프트
EXPLANATION_SYSTEM_PROMPT = """당신은 수능 수학 전문 튜터입니다.

## 출력 규칙
1. LaTeX 사용 금지 (\\frac, \\sin 등 사용 X)
2. 마크다운 수식 사용 금지
3. \\n, \\ 문자 그대로 사용 금지
4. 조합은 C(n, r) 형식으로 작성
5. 분수는 1/2 형식으로 작성
6. 곱셈은 × 기호 사용

## 출력 JSON 형식
{
    "problem_review": "[1. 문제 리뷰] 내용 (제목 미포함)",
    "condition_interpretation": "[2. 조건 해석] 내용 (제목 미포함)",
    "solution": "[3. 문제 풀이] 내용. 마지막에 반드시 '답: ②' 또는 '답: 17' 형식으로 끝낼 것"
}
"""


# === 해설 수준별 가이드라인 ===

LEVEL_GUIDELINES = {
    "초급": """
## 초급 해설 가이드라인
- 기본 개념부터 상세하게 설명
- 모든 계산 과정을 단계별로 보여주기
- 공식 사용 시 공식의 의미도 함께 설명
- 어려운 용어는 쉬운 말로 풀어서 설명
""",
    "중급": """
## 중급 해설 가이드라인
- 핵심 개념 위주로 설명
- 중요한 계산 과정만 보여주기
- 자주 사용되는 공식은 바로 적용
- 문제 해결의 핵심 아이디어 강조
""",
    "고급": """
## 고급 해설 가이드라인
- 간결하고 핵심적인 풀이
- 계산 과정은 최소화
- 고급 기법이나 빠른 풀이법 제시
- 문제의 본질적 구조 파악에 집중
"""
}


# === 과목별 핵심 개념 ===

SUBJECT_CONCEPTS = {
    "확률과통계": {
        "경우의 수": ["순열", "조합", "중복순열", "중복조합", "분할"],
        "확률": ["조건부확률", "독립사건", "종속사건", "베이즈 정리"],
        "확률분포": ["이산확률분포", "이항분포", "정규분포", "표준정규분포"],
        "통계적 추정": ["모평균 추정", "신뢰구간", "표본평균"]
    },
    "미적분": {
        "수열": ["등차수열", "등비수열", "수열의 극한", "급수"],
        "미분법": ["도함수", "미분가능성", "극대극소", "변곡점"],
        "적분법": ["부정적분", "정적분", "치환적분", "부분적분"]
    },
    "기하": {
        "이차곡선": ["포물선", "타원", "쌍곡선", "접선"],
        "평면벡터": ["벡터의 연산", "내적", "위치벡터"],
        "공간도형과 공간좌표": ["공간벡터", "직선의 방정식", "평면의 방정식"]
    }
}


# 답 형식 템플릿
ANSWER_FORMAT_TEMPLATE = """
## 답 출력 형식
- 객관식: "답: ①" 또는 "답: ②" 형식
- 주관식: "답: {숫자}" 형식 (예: "답: 17")

solution 마지막에 반드시 위 형식으로 답을 명시하세요.
"""


def build_full_prompt(
    problem_text: str,
    subject: str,
    unit: str,
    explanation_level: str,
    question_type: str
) -> str:
    """
    전체 프롬프트 구성

    Args:
        problem_text: 문제 텍스트
        subject: 과목 (확률과통계/미적분/기하)
        unit: 단원
        explanation_level: 해설 수준 (초급/중급/고급)
        question_type: 문제 유형 (objective/subjective)

    Returns:
        완성된 프롬프트 문자열
    """
    level_guide = LEVEL_GUIDELINES.get(explanation_level, LEVEL_GUIDELINES["중급"])

    prompt = f"""
## 문제
{problem_text}

## 과목: {subject}
## 단원: {unit}
## 해설 수준: {explanation_level}
## 문제 유형: {"객관식" if question_type == "objective" else "주관식"}

{level_guide}

{ANSWER_FORMAT_TEMPLATE}

위 형식에 맞춰 JSON으로 출력하세요.
"""
    return prompt
