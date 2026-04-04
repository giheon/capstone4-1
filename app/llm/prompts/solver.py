SOLVER_BASE_SYSTEM_PROMPT = """너는 수능 수학 실전 풀이를 제공하는 풀이 엔진이다.

반드시 지켜야 할 규칙:
1. solution은 수능 실전 감각의 간결한 풀이만 작성한다.
2. 장황한 이론 설명, 완전탐색식 조건 분기 나열은 피한다.
3. answer는 최종 답만 작성한다.
4. concept_ids는 반드시 제공된 concept_id 목록 안에서만 고른다.
5. flow는 실제 풀이에 사용한 concept의 순서를 concept_id 기준으로 작성한다.
6. flow의 concept_id는 concept_ids와 일관되게 작성한다.
7. action id는 사용하지 않는다.
"""


def build_solver_system_prompt(concept_reference: str) -> str:
    return (
        f"{SOLVER_BASE_SYSTEM_PROMPT}\n\n"
        "사용 가능한 concept_id 목록은 아래와 같다. 반드시 아래 목록에서만 선택하라.\n"
        f"{concept_reference}"
    )


def build_solver_user_prompt(question_text: str | None, has_image: bool) -> str:
    parts = ["다음 수능 수학 문제를 분석하고 풀이, 답, 개념, 흐름을 생성하라."]
    if question_text:
        parts.append(f"문제 텍스트:\n{question_text}")
    if has_image:
        parts.append("문제 이미지가 함께 제공된다. 이미지 내용을 함께 반영하라.")
    return "\n\n".join(parts)
