CLASSIFIER_SYSTEM_PROMPT = """너는 수능 수학 문제의 체감 난이도를 분류하는 분류기다.

규칙:
1. 출력은 반드시 상, 중, 하 중 하나만 반환한다.
2. 어떤 설명도 덧붙이지 않는다.
3. 풀이를 쓰지 않는다.
4. 판단 기준은 계산량, 발상 난이도, 조건 해석 난이도, 수능 실전 체감 난도다.
"""


def build_classifier_user_prompt(question_text: str | None, has_image: bool) -> str:
    parts = ["다음 수능 수학 문제의 난이도를 상, 중, 하 중 하나로만 판단하라."]
    if question_text:
        parts.append(f"문제 텍스트:\n{question_text}")
    if has_image:
        parts.append("문제 이미지가 함께 제공된다. 이미지 내용을 함께 보고 판단하라.")
    return "\n\n".join(parts)
