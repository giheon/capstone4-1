"""
Prompt templates for math explanation generation.
Based on 100-point example criteria from presentation.
"""

# System prompt for explanation generation
SYSTEM_PROMPT = """당신은 수능 수학 전문 튜터입니다.
학생이 스스로 문제를 풀 수 있도록 '재현 가능한 사고 과정'을 제시하는 것이 목표입니다.

## 핵심 원칙
1. 모든 조건을 빠짐없이 활용할 것
2. 각 단계에서 "왜 이 방법을 선택했는가"를 명시할 것
3. 수식은 LaTeX 형식으로 작성할 것 (예: $\\frac{1}{2}$, $\\sin\\theta$)
4. 결론 연결 시 "∴" 기호를 사용할 것

## 출력 형식
반드시 아래 JSON 형식으로 출력하세요:

{
    "section_review": "문제 리뷰 내용",
    "section_interpret": "조건 해석 내용",
    "section_solve": "문제 풀이 내용 (STEP 포함)",
    "answer": "최종 정답"
}"""


# Main explanation prompt
EXPLANATION_PROMPT = """
## 문제
{problem_text}

## 문제 유형
{problem_type}

## 해설 작성 지침

### 1. 문제 리뷰
- 문제 상황을 간결하게 요약
- 구하고자 하는 것을 명확히 명시
- "주어진 조건"과 "구해야 할 것"을 구분

### 2. 조건 해석
- 각 조건의 수학적 의미를 분석
- 조건들 간의 연결고리를 파악
- 핵심 조건과 보조 조건을 구분
- 어떤 개념/공식이 필요한지 언급

### 3. 문제 풀이
- STEP 단위로 논리적 전개
- 각 STEP에서 사용한 조건과 이유 명시
- 중간 결론은 "∴"로 연결
- 최종 답 도출 과정 명확히

### 4. 정답
- 최종 답만 간결하게

## 수식 표기 규칙
- 분수: $\\frac{{a}}{{b}}$
- 적분: $\\int_{{a}}^{{b}} f(x) dx$
- 극한: $\\lim_{{x \\to a}} f(x)$
- 삼각함수: $\\sin$, $\\cos$, $\\tan$
- 로그: $\\log$, $\\ln$
- 루트: $\\sqrt{{x}}$, $\\sqrt[n]{{x}}$

위 지침에 따라 JSON 형식으로 해설을 작성하세요.
"""


# OCR prompt for image-to-text
OCR_PROMPT = """이미지에서 수학 문제를 정확하게 텍스트로 추출하세요.

## 추출 지침
1. 문제 번호와 문제 내용을 구분
2. 수식은 LaTeX 형식으로 변환 (예: 분수는 \\frac{}{}, 적분은 \\int 등)
3. 그래프나 도형이 있으면 [그래프: 설명] 또는 [도형: 설명] 형식으로 기술
4. 보기가 있으면 번호와 함께 나열

## 출력 형식
{
    "problem_number": "문제 번호",
    "problem_text": "문제 전체 내용 (LaTeX 수식 포함)",
    "has_figure": true/false,
    "figure_description": "도형/그래프 설명 (있는 경우)",
    "choices": ["보기1", "보기2", ...] (객관식인 경우)
}
"""


# Difficulty classification prompt
DIFFICULTY_PROMPT = """다음 수학 문제의 난이도를 분류하세요.

## 문제
{problem_text}

## 난이도 기준

### 킬러 (상위 4% 이하)
- 복합적인 개념 결합 필요
- 3개 이상의 핵심 조건 활용
- 비정형적인 접근 방식 요구
- 수능 21번, 22번, 29번, 30번 수준

### 준킬러 (상위 11% 이하)
- 2-3개 개념 결합
- 표준적이지 않은 풀이 단계
- 수능 20번, 28번 수준

### 일반 (상위 11% 초과)
- 단일 개념 적용
- 정형화된 풀이 패턴
- 수능 1-19번, 23-27번 수준

## 출력 형식
{
    "difficulty": "킬러" | "준킬러" | "일반",
    "problem_type": "미적분" | "확률과통계" | "기하" | "수학1" | "수학2",
    "reasoning": "난이도 판단 이유"
}
"""


# Quality evaluation prompt for LangSmith
QUALITY_EVALUATION_PROMPT = """다음 수학 해설의 품질을 평가하세요.

## 원본 문제
{problem_text}

## 생성된 해설
{explanation}

## 평가 기준 (각 항목 0-100점)

### 1. 조건 사용 완전성 (25%)
- 문제에 주어진 모든 조건을 활용했는가
- 누락된 조건이 없는가
- 불필요한 가정을 하지 않았는가

### 2. 논리 전개 명확성 (25%)
- 각 STEP 간 연결이 자연스러운가
- 비약 없이 단계적으로 진행되는가
- 결론이 논리적으로 도출되는가

### 3. 재현 가능성 (30%)
- 학생이 이 해설만 보고 유사 문제를 풀 수 있는가
- "왜 이렇게 풀었는지"가 명확한가
- 일반화 가능한 풀이 전략이 제시되었는가

### 4. 수식 표현 정확성 (20%)
- LaTeX 문법이 올바른가
- 수식이 명확하게 표현되었는가
- 수학적 표기가 정확한가

## 출력 형식
{
    "condition_usage": 0-100,
    "logical_flow": 0-100,
    "reproducibility": 0-100,
    "latex_accuracy": 0-100,
    "total_score": 가중평균,
    "feedback": "개선이 필요한 부분",
    "pass": true/false (75점 이상이면 true)
}
"""


# Regeneration prompt when quality is below threshold
REGENERATION_PROMPT = """이전 해설이 품질 기준을 충족하지 못했습니다. 피드백을 반영하여 다시 작성하세요.

## 문제
{problem_text}

## 이전 해설의 문제점
{feedback}

## 개선 방향
{improvement_suggestions}

위 피드백을 반영하여 더 나은 해설을 JSON 형식으로 작성하세요.
"""
