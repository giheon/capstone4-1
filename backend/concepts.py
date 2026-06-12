"""
Beginner-level concept references for explanation prompts.

This module selects a small set of relevant concepts from the current OCR/routing
metadata and formats them as prompt-only reference material. The model still has
to emit the normal OutputContract JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Concept:
    subject: str
    unit: str
    concept_id: str
    name: str
    description: str


CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "공통",
        "수학 I",
        "exponent_laws",
        "지수법칙",
        r"""지수법칙은 같은 밑을 가진 거듭제곱을 계산할 때 쓰는 기본 규칙입니다.
대표 공식은 a^m a^n = a^{m+n}, \frac{a^m}{a^n}=a^{m-n}, (a^m)^n=a^{mn}입니다.
수능에서는 밑을 맞추거나 지수를 비교할 수 있는 형태로 바꿀 때 자주 사용합니다.""",
    ),
    Concept(
        "공통",
        "수학 I",
        "log_laws",
        "로그의 성질",
        r"""로그의 성질은 곱셈, 나눗셈, 거듭제곱을 덧셈과 뺄셈으로 바꾸는 규칙입니다.
대표 공식은 \log_a MN=\log_a M+\log_a N, \log_a \frac{M}{N}=\log_a M-\log_a N, \log_a M^r=r\log_a M입니다.
복잡한 로그식을 합치거나 쪼개서 조건과 연결할 때 사용합니다.""",
    ),
    Concept(
        "공통",
        "수학 I",
        "trig_definition",
        "삼각함수의 기본 정의",
        r"""삼각함수는 각에 따라 정해지는 비율입니다.
단위원에서 각 \theta에 대응하는 점이 (x,y)이면 \cos\theta=x, \sin\theta=y, \tan\theta=\frac{y}{x}입니다.
삼각함수 문제에서는 각이 어느 사분면에 있는지에 따라 부호가 달라지는 점이 중요합니다.""",
    ),
    Concept(
        "공통",
        "수학 I",
        "trig_graph",
        "삼각함수의 그래프와 주기",
        r"""\sin x와 \cos x의 주기는 2\pi이고, \tan x의 주기는 \pi입니다.
삼각함수는 일정한 간격으로 값이 반복되므로 그래프의 주기, 최대·최소, 대칭성을 이용해 해의 개수나 함수값을 판단합니다.""",
    ),
    Concept(
        "공통",
        "수학 II",
        "function_limit",
        "함수의 극한",
        r"""함수의 극한은 x가 어떤 값에 가까워질 때 f(x)가 어떤 값에 가까워지는지를 보는 개념입니다.
\lim_{x\to a} f(x)=L은 x가 a에 가까워질 때 f(x)가 L에 가까워진다는 뜻입니다.
실제 함수값 f(a)와 극한값은 다를 수 있습니다.""",
    ),
    Concept(
        "공통",
        "수학 II",
        "continuity",
        "함수의 연속",
        r"""함수 f(x)가 x=a에서 연속이려면 f(a)가 정의되고, \lim_{x\to a}f(x)가 존재하며, 그 극한값이 f(a)와 같아야 합니다.
구간별 함수나 미정계수 문제에서는 경계점의 좌극한, 우극한, 함수값을 비교합니다.""",
    ),
    Concept(
        "공통",
        "수학 II",
        "derivative_definition",
        "미분계수의 정의",
        r"""미분계수는 한 점에서의 순간변화율입니다.
f'(a)=\lim_{h\to 0}\frac{f(a+h)-f(a)}{h}이고, 그래프에서는 x=a에서 접선의 기울기를 뜻합니다.
문제의 극한식이 이 형태와 비슷하면 미분계수로 해석할 수 있는지 먼저 봅니다.""",
    ),
    Concept(
        "공통",
        "수학 II",
        "derivative_sign",
        "도함수의 부호와 증가·감소",
        r"""도함수 f'(x)의 부호는 함수 f(x)의 증가와 감소를 알려줍니다.
f'(x)>0이면 증가하고, f'(x)<0이면 감소합니다.
f'(x)=0이 되는 지점을 기준으로 부호 변화를 확인해 그래프를 해석합니다.""",
    ),
    Concept(
        "공통",
        "수학 II",
        "local_extrema",
        "극대와 극소",
        r"""극대와 극소는 함수가 주변보다 커지거나 작아지는 지점입니다.
도함수의 부호가 +에서 -로 바뀌면 극대이고, -에서 +로 바뀌면 극소입니다.
단순히 f'(x)=0이라고 해서 항상 극값인 것은 아니므로 부호 변화까지 확인해야 합니다.""",
    ),
    Concept(
        "미적분",
        "수열의 극한",
        "sequence_convergence",
        "수열의 수렴과 발산",
        r"""수열 {a_n}에서 n이 한없이 커질 때 a_n이 일정한 값 a에 가까워지면 수렴한다고 합니다.
\lim_{n\to\infty}a_n=a로 나타냅니다.
값이 한없이 커지거나 일정한 값에 가까워지지 않고 흔들리면 발산입니다.""",
    ),
    Concept(
        "미적분",
        "급수",
        "series_definition",
        "급수와 부분합",
        r"""급수는 수열의 항들을 계속 더한 것입니다.
\sum_{n=1}^{\infty}a_n의 n항까지의 부분합을 S_n이라 하면, 급수의 합은 \lim_{n\to\infty}S_n입니다.
급수 문제에서는 먼저 부분합 S_n을 잡는 것이 핵심입니다.""",
    ),
    Concept(
        "미적분",
        "여러 가지 함수의 미분",
        "trig_derivative",
        "삼각함수의 미분",
        r"""삼각함수의 기본 미분 공식은 (\sin x)'=\cos x, (\cos x)'=-\sin x, (\tan x)'=\sec^2 x입니다.
삼각함수 미분 문제에서는 미분 공식뿐 아니라 주기성과 부호를 함께 봐야 합니다.""",
    ),
    Concept(
        "미적분",
        "여러 가지 미분법",
        "chain_rule",
        "합성함수의 미분",
        r"""합성함수는 바깥 함수와 안쪽 함수를 차례로 미분합니다.
(f(g(x)))'=f'(g(x))g'(x)입니다.
복잡한 함수가 괄호 안에 들어 있으면 안쪽 함수를 먼저 확인하고, 마지막에 안쪽 함수의 미분을 곱해야 합니다.""",
    ),
    Concept(
        "미적분",
        "도함수의 활용",
        "critical_points",
        "극값 후보",
        r"""극값 후보는 보통 f'(x)=0이 되는 점과 미분 불가능한 점입니다.
후보라고 해서 무조건 극값은 아니며, 실제 극대·극소인지는 도함수의 부호가 바뀌는지 확인해야 합니다.""",
    ),
    Concept(
        "미적분",
        "도함수의 활용",
        "monotonicity_table",
        "증감표",
        r"""증감표는 도함수의 부호를 기준으로 함수가 증가하는 구간과 감소하는 구간을 정리한 표입니다.
f'(x)>0이면 증가, f'(x)<0이면 감소입니다.
극값, 최댓값·최솟값, 방정식의 실근 개수를 판단할 때 사용합니다.""",
    ),
    Concept(
        "미적분",
        "여러 가지 적분법",
        "substitution_integral",
        "치환적분",
        r"""치환적분은 복잡한 합성함수 형태의 적분을 단순하게 바꾸는 방법입니다.
안쪽 식을 u=g(x)로 두면 du=g'(x)\,dx가 됩니다.
적분식 안에 어떤 식과 그 식의 미분이 함께 보이면 치환적분을 먼저 생각합니다.""",
    ),
    Concept(
        "미적분",
        "정적분의 활용",
        "area_between_curves",
        "두 곡선 사이의 넓이",
        r"""두 곡선 사이의 넓이는 위쪽 함수에서 아래쪽 함수를 뺀 값을 적분합니다.
\int_a^b \{f(x)-g(x)\}\,dx 형태로 계산합니다.
구간 중간에 위아래가 바뀌면 교점을 기준으로 구간을 나누어야 합니다.""",
    ),
    Concept(
        "확률과통계",
        "여러 가지 순열",
        "permutation_with_repetition",
        "같은 것이 있는 순열",
        r"""같은 것이 포함된 n개를 일렬로 배열할 때 같은 것끼리의 순서를 중복으로 세면 안 됩니다.
같은 것이 각각 p,q,r,\dots개씩 있으면 경우의 수는 \frac{n!}{p!q!r!\cdots}입니다.""",
    ),
    Concept(
        "확률과통계",
        "중복조합과 이항정리",
        "combination",
        "조합",
        r"""조합은 서로 다른 n개에서 순서를 생각하지 않고 r개를 고르는 경우의 수입니다.
{}_nC_r=\frac{n!}{r!(n-r)!}입니다.
순서를 따지면 순열, 순서를 따지지 않으면 조합입니다.""",
    ),
    Concept(
        "확률과통계",
        "확률의 뜻과 활용",
        "probability_basic",
        "확률의 기본",
        r"""확률은 전체 경우 중 원하는 경우가 차지하는 비율입니다.
모든 경우가 똑같이 일어날 가능성이 있을 때 P(A)=\frac{n(A)}{n(S)}입니다.
먼저 표본공간 S와 사건 A를 정확히 구분해야 합니다.""",
    ),
    Concept(
        "확률과통계",
        "조건부확률",
        "conditional_probability",
        "조건부확률",
        r"""조건부확률은 어떤 사건이 이미 일어났다는 조건 아래에서 다른 사건이 일어날 확률입니다.
P(A\mid B)=\frac{P(A\cap B)}{P(B)}입니다.
조건이 주어지면 표본공간이 전체 S가 아니라 B로 줄어든다고 생각하면 됩니다.""",
    ),
    Concept(
        "확률과통계",
        "이산확률변수의 확률분포",
        "expected_value",
        "기댓값",
        r"""기댓값은 확률변수의 평균적인 값을 의미합니다.
E(X)=\sum x_iP(X=x_i)입니다.
각 값에 그 값이 나올 확률을 곱해서 모두 더합니다.""",
    ),
    Concept(
        "확률과통계",
        "연속확률변수의 확률분포",
        "normal_distribution",
        "정규분포",
        r"""정규분포는 평균을 중심으로 좌우 대칭인 종 모양의 분포입니다.
X\sim N(m,\sigma^2)이면 평균은 m, 표준편차는 \sigma입니다.
정규분포 문제에서는 표준화해서 표준정규분포로 바꾸는 것이 핵심입니다.""",
    ),
    Concept(
        "기하",
        "포물선",
        "parabola_definition",
        "포물선의 정의",
        r"""포물선은 한 점인 초점과 한 직선인 준선까지의 거리가 같은 점들의 집합입니다.
포물선 위의 점 P에 대하여 초점이 F, 준선에 내린 수선의 발이 H이면 PF=PH입니다.
이 거리 같음을 길이 조건으로 바꾸는 것이 중요합니다.""",
    ),
    Concept(
        "기하",
        "타원",
        "ellipse_definition",
        "타원의 정의",
        r"""타원은 두 초점으로부터의 거리의 합이 일정한 점들의 집합입니다.
타원 위의 점 P에 대해 PF+PF'=2a입니다.
타원 문제에서는 거리합 조건을 먼저 찾아야 합니다.""",
    ),
    Concept(
        "기하",
        "쌍곡선",
        "hyperbola_definition",
        "쌍곡선의 정의",
        r"""쌍곡선은 두 초점으로부터의 거리의 차가 일정한 점들의 집합입니다.
쌍곡선 위의 점 P에 대해 |PF-PF'|=2a입니다.
쌍곡선 문제에서는 거리의 합이 아니라 차를 본다는 점이 타원과 다릅니다.""",
    ),
    Concept(
        "기하",
        "벡터의 연산",
        "vector_addition",
        "벡터의 덧셈과 뺄셈",
        r"""벡터의 덧셈은 이동을 이어 붙이는 것입니다.
\vec{AB}+\vec{BC}=\vec{AC}입니다.
벡터의 뺄셈은 같은 시작점에서 두 끝점을 비교하는 것으로 볼 수 있습니다.""",
    ),
    Concept(
        "기하",
        "벡터의 내적",
        "dot_product_definition",
        "벡터의 내적",
        r"""두 벡터 \vec a,\vec b의 내적은 \vec a\cdot\vec b=|\vec a||\vec b|\cos\theta입니다.
내적은 두 벡터가 같은 방향으로 얼마나 겹치는지를 나타냅니다.
각도, 길이, 수직 조건을 내적으로 바꿀 때 자주 사용합니다.""",
    ),
    Concept(
        "기하",
        "공간도형",
        "orthogonal_projection",
        "정사영",
        r"""정사영은 도형을 어떤 평면이나 직선에 수직 방향으로 내려찍은 그림입니다.
공간도형 문제에서는 입체를 그대로 보지 말고 필요한 평면에 정사영해서 평면도형 문제로 바꿉니다.""",
    ),
    Concept(
        "기하",
        "공간좌표",
        "distance_3d",
        "좌표공간에서 두 점 사이의 거리",
        r"""두 점 A(x_1,y_1,z_1), B(x_2,y_2,z_2) 사이의 거리는 \sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}입니다.
공간좌표에서 거리 조건이 나오면 이 공식을 먼저 적용합니다.""",
    ),
)


def _build_concept_db() -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    concept_db: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for concept in CONCEPTS:
        concept_db.setdefault(concept.subject, {}).setdefault(concept.unit, {})[
            concept.concept_id
        ] = {
            "name": concept.name,
            "description": concept.description,
        }
    return concept_db


CONCEPT_DB = _build_concept_db()


TOPIC_TO_CONCEPT_IDS: dict[str, tuple[str, ...]] = {
    "function_limits": ("function_limit",),
    "function_continuity": ("continuity",),
    "derivative_definition": ("derivative_definition",),
    "derivative_applications_basic_1": ("derivative_sign", "local_extrema"),
    "derivative_applications_basic_2": ("critical_points", "monotonicity_table"),
    "derivative_applications_advanced": ("critical_points", "monotonicity_table", "local_extrema"),
    "derivatives_of_various_functions": ("trig_derivative",),
    "advanced_differentiation_rules": ("chain_rule",),
    "indefinite_and_definite_integrals": ("substitution_integral",),
    "advanced_integration_methods": ("substitution_integral",),
    "definite_integral_applications_basic": ("area_between_curves",),
    "definite_integral_applications_advanced": ("area_between_curves",),
    "sequence_limits": ("sequence_convergence",),
    "infinite_series": ("series_definition",),
    "probability_statistics": ("probability_basic", "conditional_probability", "expected_value"),
    "geometry": ("vector_addition", "dot_product_definition", "orthogonal_projection"),
}

UNIT_TO_CONCEPT_IDS: dict[str, tuple[str, ...]] = {
    "미분법": ("trig_derivative", "chain_rule", "derivative_sign", "local_extrema", "critical_points"),
    "적분법": ("substitution_integral", "area_between_curves"),
    "수열": ("sequence_convergence", "series_definition"),
    "수열의 극한": ("sequence_convergence",),
    "급수": ("series_definition",),
    "여러 가지 함수의 미분": ("trig_derivative",),
    "여러 가지 미분법": ("chain_rule",),
    "도함수의 활용": ("derivative_sign", "local_extrema", "critical_points", "monotonicity_table"),
    "여러 가지 적분법": ("substitution_integral",),
    "정적분의 활용": ("area_between_curves",),
    "여러 가지 순열": ("permutation_with_repetition",),
    "중복조합과 이항정리": ("combination",),
    "확률의 뜻과 활용": ("probability_basic",),
    "조건부확률": ("conditional_probability",),
    "이산확률변수의 확률분포": ("expected_value",),
    "연속확률변수의 확률분포": ("normal_distribution",),
    "포물선": ("parabola_definition",),
    "타원": ("ellipse_definition",),
    "쌍곡선": ("hyperbola_definition",),
    "벡터의 연산": ("vector_addition",),
    "벡터의 내적": ("dot_product_definition",),
    "공간도형": ("orthogonal_projection",),
    "공간좌표": ("distance_3d",),
}


def _normalize_subject(subject: str) -> str:
    if subject in {"확률과 통계", "확률과통계"}:
        return "확률과통계"
    return subject


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_concepts(
    subject: str,
    unit: str = "",
    curriculum_area: str = "",
    major_topics: list[str] | None = None,
    limit: int = 6,
) -> list[Concept]:
    """Select a compact, deterministic concept subset for the current problem."""
    normalized_subject = _normalize_subject(subject)
    requested_ids: list[str] = []

    for topic in major_topics or []:
        requested_ids.extend(TOPIC_TO_CONCEPT_IDS.get(topic, ()))
    requested_ids.extend(TOPIC_TO_CONCEPT_IDS.get(curriculum_area, ()))
    requested_ids.extend(UNIT_TO_CONCEPT_IDS.get(unit, ()))
    requested_ids = _unique(requested_ids)

    selected: list[Concept] = []
    for concept_id in requested_ids:
        for concept in CONCEPTS:
            if concept.concept_id != concept_id:
                continue
            if concept.subject in {normalized_subject, "공통"}:
                selected.append(concept)
                break

    if len(selected) < limit:
        for concept in CONCEPTS:
            if concept in selected:
                continue
            if concept.subject == normalized_subject and (not unit or concept.unit == unit):
                selected.append(concept)
            if len(selected) >= limit:
                break

    if len(selected) < limit and normalized_subject == "미적분":
        for concept_id in ("function_limit", "continuity", "derivative_definition", "derivative_sign", "local_extrema"):
            for concept in CONCEPTS:
                if concept.concept_id == concept_id and concept not in selected:
                    selected.append(concept)
                    break
            if len(selected) >= limit:
                break

    return selected[:limit]


def build_concept_reference_prompt(
    subject: str,
    unit: str = "",
    curriculum_area: str = "",
    major_topics: list[str] | None = None,
) -> str:
    concepts = select_concepts(
        subject=subject,
        unit=unit,
        curriculum_area=curriculum_area,
        major_topics=major_topics,
    )
    if not concepts:
        return ""

    concept_lines: list[str] = []
    for index, concept in enumerate(concepts, start=1):
        concept_lines.append(
            f"{index}. {concept.name} ({concept.concept_id}, {concept.subject} > {concept.unit})\n"
            f"{concept.description.strip()}"
        )

    return (
        "<ConceptReference>\n"
        "아래 개념 집합은 초급 해설의 concept_explanation을 작성할 때만 참고한다.\n"
        "문제와 직접 관련된 개념만 골라 학생에게 필요한 만큼 쉽게 풀어 설명하라.\n"
        "아래 문장을 그대로 복사하지 말고, 현재 문제의 조건과 풀이 흐름에 맞게 재구성하라.\n"
        "수식을 출력 JSON에 넣을 때는 반드시 OutputContract의 latex block으로 분리하고, "
        "delimiter($, $$, \\(\\), \\[\\]) 없이 순수 LaTeX body만 작성하라.\n"
        "이 ConceptReference 태그와 개념 ID는 출력 JSON에 절대 포함하지 마라.\n\n"
        + "\n\n".join(concept_lines)
        + "\n</ConceptReference>"
    )
