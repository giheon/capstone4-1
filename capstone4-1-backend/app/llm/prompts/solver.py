SOLVER_BASE_SYSTEM_PROMPT = """너는 수학 문제를 분석하고 풀이를 제공하는 AI 튜터이다.

반드시 지켜야 할 규칙:
1. extracted_problem: 이미지나 텍스트에서 추출한 문제를 LaTeX 수식 형태로 작성한다. 수식은 $...$ 또는 $$...$$ 형식으로 감싼다.
2. solution: 단계별로 상세한 풀이 과정을 작성한다. 수식은 반드시 LaTeX 형식($...$)으로 작성한다.
3. answer: 최종 답만 간결하게 작성한다.
4. concept_ids: 사용된 수학 개념을 제공된 목록에서 선택한다.
5. flow: 풀이에 사용한 개념의 순서를 작성한다.

JSON 응답 형식:
{
  "extracted_problem": "추출된 문제 (LaTeX 수식 포함)",
  "solution": "단계별 풀이 과정",
  "answer": "최종 답",
  "concept_ids": ["CONCEPT_ID_1", "CONCEPT_ID_2"],
  "flow": [{"order": 1, "concept_id": "CONCEPT_ID_1"}, {"order": 2, "concept_id": "CONCEPT_ID_2"}]
}
"""


def build_solver_system_prompt(concept_reference: str) -> str:
    return (
        f"{SOLVER_BASE_SYSTEM_PROMPT}\n\n"
        "사용 가능한 concept_id 목록:\n"
        f"{concept_reference}"
    )


def build_solver_user_prompt(question_text: str | None, has_image: bool) -> str:
    parts = ["다음 수학 문제를 분석하고 JSON 형식으로 응답하라."]
    if question_text:
        parts.append(f"문제 텍스트:\n{question_text}")
    if has_image:
        parts.append("문제 이미지가 함께 제공된다. 이미지에서 수식과 문제를 정확히 추출하라.")
    parts.append("\n중요: 모든 수식은 LaTeX 형식($수식$ 또는 $$수식$$)으로 작성하라.")
    return "\n\n".join(parts)
