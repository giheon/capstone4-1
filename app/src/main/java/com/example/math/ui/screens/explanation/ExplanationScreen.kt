package com.example.math.ui.screens.explanation

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.math.ui.components.LatexFormula
import com.example.math.ui.theme.*
import kotlinx.coroutines.delay

// 콘텐츠 타입 정의
sealed class ContentItem {
    data class Text(val text: String) : ContentItem()
    data class Formula(val latex: String) : ContentItem()  // LaTeX 형식
    data class Hint(val text: String) : ContentItem()  // 부드러운 강조
    data class Answer(val text: String) : ContentItem()  // 정답 (문제풀이 내부)
}

// 전체 해설 데이터
data class ExplanationData(
    val section1: List<ContentItem>,
    val section2: List<ContentItem>,
    val steps: List<Pair<String, List<ContentItem>>>,  // (Step 제목, 내용)
)

@Composable
fun StreamingText(
    text: String,
    modifier: Modifier = Modifier,
    charDelayMs: Long = 20L,
    color: Color = Gray600,
    fontSize: androidx.compose.ui.unit.TextUnit = 14.sp,
    fontWeight: FontWeight = FontWeight.Normal,
    lineHeight: androidx.compose.ui.unit.TextUnit = 22.sp,
    onComplete: () -> Unit = {}
) {
    var charCount by remember { mutableIntStateOf(0) }
    val displayedText by remember(text) {
        derivedStateOf { text.take(charCount) }
    }

    LaunchedEffect(text) {
        charCount = 0
        for (i in 1..text.length) {
            delay(charDelayMs)
            charCount = i
        }
        onComplete()
    }

    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = Alignment.TopStart
    ) {
        Text(
            text = text,
            color = Color.Transparent,
            fontSize = fontSize,
            fontWeight = fontWeight,
            lineHeight = lineHeight
        )
        Text(
            text = displayedText,
            color = color,
            fontSize = fontSize,
            fontWeight = fontWeight,
            lineHeight = lineHeight
        )
    }
}

// LaTeX 수식 박스
@Composable
fun FormulaBox(
    formula: String,
    onComplete: () -> Unit = {}
) {
    LatexFormula(
        latex = formula,
        onRendered = onComplete
    )
}

// 힌트/안내 박스 (부드러운 파란색)
@Composable
fun HintBox(
    text: String,
    onComplete: () -> Unit = {}
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Blue50.copy(alpha = 0.7f))
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        StreamingText(
            text = text,
            color = Blue900.copy(alpha = 0.85f),
            fontSize = 13.sp,
            fontWeight = FontWeight.Normal,
            lineHeight = 20.sp,
            charDelayMs = 20L,
            onComplete = onComplete
        )
    }
}

// 정답 표시 (문제풀이 내부)
@Composable
fun AnswerText(
    text: String,
    onComplete: () -> Unit = {}
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Blue500.copy(alpha = 0.1f))
            .padding(horizontal = 14.dp, vertical = 12.dp)
    ) {
        StreamingText(
            text = text,
            color = Blue600,
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            lineHeight = 20.sp,
            charDelayMs = 25L,
            onComplete = onComplete
        )
    }
}

// 섹션 라벨 (번호만 파란 박스, 제목은 옆에)
@Composable
fun SectionLabel(number: String, title: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(4.dp))
                .background(Blue500)
                .padding(horizontal = 8.dp, vertical = 2.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = number,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = title,
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            color = Gray600
        )
    }
}

// 수준별 해설 데이터 생성
private fun getExplanationDataForLevel(level: String): ExplanationData {
    return when (level) {
        "초급" -> ExplanationData(
            section1 = listOf(
                ContentItem.Text("안녕하세요! 이 문제를 함께 풀어볼게요. 먼저 문제를 천천히 살펴봅시다."),
                ContentItem.Text("이 문제는 삼각함수가 포함된 합성함수 형태예요. 합성함수란 함수 안에 또 다른 함수가 들어있는 것을 말해요."),
                ContentItem.Text("sin(ax + b + sin x)처럼 sin 안에 또 다른 sin이 들어있죠? 이런 문제는 한 번에 풀려고 하면 어려워요."),
                ContentItem.Hint("핵심 전략: 먼저 특별한 x값(0, 2π 등)을 대입해서 a, b의 값을 찾고, 그 다음에 극대점을 찾는 순서로 풀어요!")
            ),
            section2 = listOf(
                ContentItem.Text("(가) 조건부터 살펴볼게요. x = 0을 넣으면 어떻게 될까요?"),
                ContentItem.Text("sin 0 = 0이므로, 안쪽 식에서 sin x 부분이 0이 되어요."),
                ContentItem.Formula("f(0) = \\sin(a \\cdot 0 + b + \\sin 0) = \\sin b"),
                ContentItem.Text("그리고 f(0) = 0이라고 했으니까:"),
                ContentItem.Formula("\\sin b = 0 \\;\\Rightarrow\\; b = n\\pi \\;\\text{(n은 정수)}"),
                ContentItem.Text("이번엔 x = 2π를 넣어볼게요. sin 2π = 0이에요."),
                ContentItem.Formula("f(2\\pi) = \\sin(2\\pi a + b)"),
                ContentItem.Text("그리고 f(2π) = 2πa + b라고 했으니까, sin X = X가 되려면 X = 0이어야 해요."),
                ContentItem.Formula("2\\pi a + b = 0 \\;\\Rightarrow\\; b = -2\\pi a"),
                ContentItem.Text("b = nπ이고 b = -2πa를 합치면:"),
                ContentItem.Formula("a = \\frac{n}{2}"),
                ContentItem.Text("1 ≤ a ≤ 2 조건에서 a = 1, 3/2, 2가 가능해요."),
                ContentItem.Hint("(나) 조건으로 후보를 걸러내면 a = 3/2, b = -3π만 남아요!")
            ),
            steps = listOf(
                "도함수 구하기" to listOf(
                    ContentItem.Text("f(x) = sin(ax + b + sin x)의 도함수를 구해볼게요."),
                    ContentItem.Text("합성함수의 미분 공식 (sin u)' = cos u · u'를 사용해요."),
                    ContentItem.Formula("f'(x) = \\cos(ax + b + \\sin x) \\cdot (a + \\cos x)"),
                    ContentItem.Text("a = 3/2를 대입하면:"),
                    ContentItem.Formula("f'(x) = \\cos\\left(\\frac{3x}{2} - 3\\pi + \\sin x\\right) \\cdot \\left(\\frac{3}{2} + \\cos x\\right)"),
                    ContentItem.Hint("cos x의 최솟값은 -1이니까, 3/2 + cos x ≥ 1/2 > 0이에요. 항상 양수!")
                ),
                "극대점 찾기" to listOf(
                    ContentItem.Text("f'(x) = 0이 되려면 cos(...) = 0이어야 해요."),
                    ContentItem.Text("g(x) = 3x/2 - 3π + sin x라고 하면, cos(g(x)) = 0이 되는 점을 찾아요."),
                    ContentItem.Text("코사인이 0이 되는 값: ±π/2, ±3π/2, ±5π/2, ..."),
                    ContentItem.Formula("g(0) = -3\\pi, \\quad g(4\\pi) = 3\\pi"),
                    ContentItem.Text("g(x)가 -3π에서 3π까지 증가하면서 코사인이 0이 되는 점이 6개예요."),
                    ContentItem.Hint("이 중 극대점은 cos가 양수→음수로 바뀌는 3개! n = 3")
                ),
                "최종 답 계산" to listOf(
                    ContentItem.Text("가장 작은 극대점 α₁은 g(x) = -3π/2가 되는 점이에요."),
                    ContentItem.Formula("\\frac{3\\alpha_1}{2} - 3\\pi + \\sin\\alpha_1 = -\\frac{3\\pi}{2}"),
                    ContentItem.Text("α₁ = π를 대입해보면:"),
                    ContentItem.Formula("\\frac{3\\pi}{2} - 3\\pi + 0 = -\\frac{3\\pi}{2} \\; \\checkmark"),
                    ContentItem.Text("정답 계산:"),
                    ContentItem.Formula("n\\alpha_1 - ab = 3\\pi - \\frac{3}{2}(-3\\pi) = 3\\pi + \\frac{9\\pi}{2} = \\frac{15\\pi}{2}"),
                    ContentItem.Formula("\\frac{q}{p}\\pi = \\frac{15}{2}\\pi \\;\\Rightarrow\\; p = 2, \\; q = 15"),
                    ContentItem.Answer("따라서 정답은 p + q = 17 입니다.")
                )
            )
        )

        "중급" -> ExplanationData(
            section1 = listOf(
                ContentItem.Text("이 문제는 처음 보면 삼각함수 안에 또 삼각함수가 들어 있는 합성함수 형태예요. 이런 문제는 식을 무작정 전개하려고 하면 오히려 길이 꼬이기 쉽습니다."),
                ContentItem.Text("그래서 먼저 겉에 있는 sin과 안쪽 식 ax + b + sin x를 분리해서 보는 게 좋아요."),
                ContentItem.Text("또 눈에 띄는 건 0, 2π, 4π 같은 삼각함수의 주기와 딱 맞는 값들이 계속 나온다는 점입니다."),
                ContentItem.Hint("먼저 특수한 x값들을 넣어서 a, b의 후보를 줄이고, 도함수 조건으로 후보를 걸러낸 뒤, 극대점의 개수와 가장 작은 극대점 α₁을 찾는 흐름으로 진행합니다.")
            ),
            section2 = listOf(
                ContentItem.Text("먼저 (가) 조건을 볼게요. x = 0, x = 2π를 준 건 삼각함수의 특수값을 이용해서 a, b를 강하게 묶으라는 신호입니다."),
                ContentItem.Text("x = 0을 넣으면 안쪽 식은 b만 남습니다."),
                ContentItem.Formula("f(0) = \\sin b = 0 \\;\\Rightarrow\\; b = n\\pi"),
                ContentItem.Text("이번에는 x = 2π를 봅니다. sin 2π = 0이라서 안쪽 식이 2πa + b로 깔끔해져요."),
                ContentItem.Formula("\\sin X = X \\;\\Rightarrow\\; X = 0 \\;\\Rightarrow\\; 2\\pi a + b = 0"),
                ContentItem.Text("두 정보를 합치면 b = −2πa이고, b는 π의 정수배여야 합니다."),
                ContentItem.Formula("a = \\frac{p}{2}, \\quad b = -p\\pi"),
                ContentItem.Text("1 ≤ a ≤ 2를 적용하면 2 ≤ p ≤ 4이므로 가능한 p는 2, 3, 4뿐이에요."),
                ContentItem.Text("(나) 조건에서 도함수를 구하면:"),
                ContentItem.Formula("f'(x) = \\cos(ax + b + \\sin x) \\cdot (a + \\cos x)"),
                ContentItem.Hint("분석 결과, 짝수 후보는 모두 탈락하고 p = 3만 남습니다. 따라서 a = 3/2, b = −3π로 확정됩니다.")
            ),
            steps = listOf(
                "확정된 값 대입" to listOf(
                    ContentItem.Text("a = 3/2, b = −3π이므로 도함수는:"),
                    ContentItem.Formula("f'(x) = \\cos\\left(\\frac{3x}{2} - 3\\pi + \\sin x\\right) \\cdot \\left(\\frac{3}{2} + \\cos x\\right)"),
                    ContentItem.Text("여기서 3/2 + cos x는 항상 양수입니다. cos x의 최솟값이 −1이므로 3/2 + cos x ≥ 1/2 > 0이기 때문이에요."),
                    ContentItem.Hint("따라서 도함수의 부호는 앞의 코사인 부분이 결정합니다.")
                ),
                "극대 후보 찾기" to listOf(
                    ContentItem.Text("g(x) = 3x/2 − 3π + sin x라고 두면:"),
                    ContentItem.Formula("g(0) = -3\\pi, \\quad g(4\\pi) = 3\\pi"),
                    ContentItem.Text("g′(x) = 3/2 + cos x > 0이므로 g(x)는 계속 증가합니다. x가 0에서 4π까지 움직이는 동안 g(x)는 −3π에서 3π까지 쭉 증가해요.")
                ),
                "극대점 개수 계산" to listOf(
                    ContentItem.Text("극값은 f′(x) = 0, 즉 cos(g(x)) = 0이 되는 지점입니다."),
                    ContentItem.Text("−3π부터 3π 사이에서 코사인이 0이 되는 값은:"),
                    ContentItem.Formula("-\\frac{5\\pi}{2}, -\\frac{3\\pi}{2}, -\\frac{\\pi}{2}, \\frac{\\pi}{2}, \\frac{3\\pi}{2}, \\frac{5\\pi}{2}"),
                    ContentItem.Text("g(x)가 증가하므로 cos(g(x))가 양수에서 음수로 바뀌는 지점이 극대입니다."),
                    ContentItem.Hint("극대점은 총 3개이고, n = 3입니다.")
                ),
                "최종 계산" to listOf(
                    ContentItem.Text("가장 작은 극대점은 g(x) = −3π/2가 되는 지점입니다."),
                    ContentItem.Formula("\\frac{3\\alpha_1}{2} - 3\\pi + \\sin\\alpha_1 = -\\frac{3\\pi}{2}"),
                    ContentItem.Text("정리하면:"),
                    ContentItem.Formula("\\frac{3\\alpha_1}{2} + \\sin\\alpha_1 = \\frac{3\\pi}{2}"),
                    ContentItem.Text("α₁ = π를 대입하면 3π/2 + sin π = 3π/2가 성립하므로 α₁ = π입니다."),
                    ContentItem.Text("구한 값을 정리하면:"),
                    ContentItem.Formula("n = 3, \\; \\alpha_1 = \\pi, \\; a = \\frac{3}{2}, \\; b = -3\\pi"),
                    ContentItem.Text("따라서:"),
                    ContentItem.Formula("n\\alpha_1 - ab = 3\\pi - \\frac{3}{2} \\cdot (-3\\pi) = 3\\pi + \\frac{9\\pi}{2} = \\frac{15\\pi}{2}"),
                    ContentItem.Text("문제에서 이 값이 (q/p)π라고 했으므로:"),
                    ContentItem.Formula("\\frac{q}{p} = \\frac{15}{2} \\;\\Rightarrow\\; p = 2, \\; q = 15"),
                    ContentItem.Answer("따라서 정답은 p + q = 17 입니다.")
                )
            )
        )

        "고급" -> ExplanationData(
            section1 = listOf(
                ContentItem.Text("합성함수 f(x) = sin(ax + b + sin x)에서 특수값 대입으로 a, b를 결정하고, 극대점 개수를 구하는 문제입니다."),
                ContentItem.Hint("접근: (가)로 a, b 후보 → (나)로 필터링 → 극대점 개수 및 α₁ 계산")
            ),
            section2 = listOf(
                ContentItem.Text("(가) 조건에서:"),
                ContentItem.Formula("f(0) = \\sin b = 0 \\;\\Rightarrow\\; b = n\\pi"),
                ContentItem.Formula("f(2\\pi) = 2\\pi a + b \\;\\Rightarrow\\; 2\\pi a + b = 0"),
                ContentItem.Text("따라서 b = −2πa, a = n/2 (n은 정수)"),
                ContentItem.Text("1 ≤ a ≤ 2에서 a ∈ {1, 3/2, 2}"),
                ContentItem.Formula("f'(x) = \\cos(ax + b + \\sin x)(a + \\cos x)"),
                ContentItem.Hint("(나) 조건 검증 시 a = 3/2, b = −3π만 만족")
            ),
            steps = listOf(
                "극대점 분석" to listOf(
                    ContentItem.Text("g(x) = 3x/2 − 3π + sin x로 치환"),
                    ContentItem.Formula("g(0) = -3\\pi, \\; g(4\\pi) = 3\\pi, \\; g'(x) > 0"),
                    ContentItem.Text("cos(g(x)) = 0인 점: g(x) = ±π/2, ±3π/2, ±5π/2"),
                    ContentItem.Text("극대: g(x) = −3π/2, π/2, 5π/2 → n = 3")
                ),
                "최종 계산" to listOf(
                    ContentItem.Formula("g(\\alpha_1) = -\\frac{3\\pi}{2} \\;\\Rightarrow\\; \\alpha_1 = \\pi"),
                    ContentItem.Formula("n\\alpha_1 - ab = 3\\pi + \\frac{9\\pi}{2} = \\frac{15\\pi}{2}"),
                    ContentItem.Formula("p = 2, \\; q = 15"),
                    ContentItem.Answer("정답: p + q = 17")
                )
            )
        )

        else -> getExplanationDataForLevel("중급")
    }
}

@Composable
fun ExplanationScreen(
    explanationLevel: String = "중급",
    imageBase64: String? = null,  // 촬영된 이미지 (추후 API 전송용)
    onReset: () -> Unit
) {
    // TODO: imageBase64를 백엔드 API로 전송하여 실제 해설 받기
    // 현재는 데모 데이터 사용
    var currentIndex by remember { mutableIntStateOf(0) }

    // 수준별 해설 데이터
    val explanationData = remember(explanationLevel) {
        getExplanationDataForLevel(explanationLevel)
    }

    // 인덱스 계산
    val section1End = explanationData.section1.size
    val section2Start = section1End
    val section2End = section2Start + 1 + explanationData.section2.size

    val stepIndices = remember(explanationData) {
        val indices = mutableListOf<Pair<Int, Int>>()
        var currentStart = section2End
        explanationData.steps.forEach { (_, items) ->
            val stepEnd = currentStart + 1 + items.size
            indices.add(Pair(currentStart, stepEnd))
            currentStart = stepEnd
        }
        indices
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
            .statusBarsPadding()
    ) {
        // 헤더 (뒤로가기 + 수준 표시)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            FilledIconButton(
                onClick = onReset,
                modifier = Modifier.size(40.dp),
                shape = CircleShape,
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = Color.White,
                    contentColor = Gray600
                )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "뒤로가기",
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            // 해설 수준 뱃지
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = Blue500.copy(alpha = 0.1f)
            ) {
                Text(
                    text = "$explanationLevel 해설",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium,
                    color = Blue600,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                )
            }
        }

        // 컨텐츠
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // [1. 문제 리뷰] - 흰색 카드
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    SectionContent(
                        number = "1",
                        title = "문제 리뷰",
                        items = explanationData.section1,
                        startIndex = 0,
                        currentIndex = currentIndex,
                        onItemComplete = { currentIndex++ }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // [2. 조건 해석] - 흰색 카드
            AnimatedVisibility(
                visible = currentIndex >= section1End,
                enter = fadeIn()
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        SectionContent(
                            number = "2",
                            title = "조건 해석",
                            items = explanationData.section2,
                            startIndex = section2Start + 1,
                            currentIndex = currentIndex,
                            onItemComplete = { currentIndex++ },
                            onHeaderShow = { currentIndex++ }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // [3. 문제 풀이] - 흰색 카드
            AnimatedVisibility(
                visible = currentIndex >= section2End,
                enter = fadeIn()
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        StepsContent(
                            steps = explanationData.steps,
                            stepIndices = stepIndices,
                            currentIndex = currentIndex,
                            onItemComplete = { currentIndex++ }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(40.dp))
        }
    }
}

@Composable
private fun SectionContent(
    number: String,
    title: String,
    items: List<ContentItem>,
    startIndex: Int,
    currentIndex: Int,
    onItemComplete: () -> Unit,
    onHeaderShow: (() -> Unit)? = null
) {
    LaunchedEffect(Unit) {
        onHeaderShow?.invoke()
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        // 섹션 라벨 (번호만 파란 박스)
        SectionLabel(number = number, title = title)

        Spacer(modifier = Modifier.height(12.dp))

        // Content Items
        items.forEachIndexed { index, item ->
            val itemIndex = startIndex + index
            AnimatedVisibility(
                visible = currentIndex >= itemIndex,
                enter = fadeIn()
            ) {
                Column {
                    when (item) {
                        is ContentItem.Text -> {
                            StreamingText(
                                text = item.text,
                                color = Gray600,
                                modifier = Modifier.padding(bottom = 8.dp),
                                onComplete = onItemComplete
                            )
                        }
                        is ContentItem.Formula -> {
                            FormulaBox(
                                formula = item.latex,
                                onComplete = onItemComplete
                            )
                        }
                        is ContentItem.Hint -> {
                            HintBox(
                                text = item.text,
                                onComplete = onItemComplete
                            )
                        }
                        is ContentItem.Answer -> {
                            AnswerText(
                                text = item.text,
                                onComplete = onItemComplete
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StepsContent(
    steps: List<Pair<String, List<ContentItem>>>,
    stepIndices: List<Pair<Int, Int>>,
    currentIndex: Int,
    onItemComplete: () -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        // 섹션 라벨 (번호만 파란 박스)
        SectionLabel(number = "3", title = "문제 풀이")

        Spacer(modifier = Modifier.height(12.dp))

        // Steps
        steps.forEachIndexed { stepIdx, (stepTitle, stepItems) ->
            val (stepStart, _) = stepIndices[stepIdx]

            AnimatedVisibility(
                visible = currentIndex >= stepStart,
                enter = fadeIn()
            ) {
                LaunchedEffect(Unit) {
                    onItemComplete()
                }

                Column(modifier = Modifier.fillMaxWidth()) {
                    if (stepIdx > 0) {
                        Spacer(modifier = Modifier.height(16.dp))
                        HorizontalDivider(
                            modifier = Modifier.padding(vertical = 8.dp),
                            color = Gray200,
                            thickness = 0.5.dp
                        )
                    }

                    // STEP 라벨만 표시 (숫자 배지 제거)
                    Text(
                        text = "STEP ${stepIdx + 1}. $stepTitle",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Blue600,
                        modifier = Modifier.padding(bottom = 10.dp)
                    )

                    // Step content items
                    stepItems.forEachIndexed { itemIdx, item ->
                        val itemIndex = stepStart + 1 + itemIdx
                        AnimatedVisibility(
                            visible = currentIndex >= itemIndex,
                            enter = fadeIn()
                        ) {
                            when (item) {
                                is ContentItem.Text -> {
                                    StreamingText(
                                        text = item.text,
                                        color = Gray600,
                                        modifier = Modifier.padding(bottom = 6.dp),
                                        onComplete = onItemComplete
                                    )
                                }
                                is ContentItem.Formula -> {
                                    FormulaBox(
                                        formula = item.latex,
                                        onComplete = onItemComplete
                                    )
                                }
                                is ContentItem.Hint -> {
                                    HintBox(
                                        text = item.text,
                                        onComplete = onItemComplete
                                    )
                                }
                                is ContentItem.Answer -> {
                                    AnswerText(
                                        text = item.text,
                                        onComplete = onItemComplete
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun ExplanationScreenPreview() {
    MathAITheme {
        ExplanationScreen(
            explanationLevel = "중급",
            onReset = {}
        )
    }
}
