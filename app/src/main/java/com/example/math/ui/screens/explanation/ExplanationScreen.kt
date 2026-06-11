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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.math.data.api.ExplanationBlock as ApiExplanationBlock
import com.example.math.data.api.ExplanationResponse
import com.example.math.data.api.InlineSpan as ApiInlineSpan
import com.example.math.ui.components.LatexFormula
import com.example.math.ui.components.LatexParagraph
import com.example.math.ui.components.LatexParagraphSpan
import com.example.math.ui.theme.*
import kotlinx.coroutines.delay

// 콘텐츠 타입 정의
sealed class ContentItem {
    data class Text(val text: String) : ContentItem()
    data class Formula(
        val latex: String,
        val display: Boolean = false
    ) : ContentItem()  // LaTeX 형식
    data class RichLine(val parts: List<InlinePart>) : ContentItem()
    data class Hint(val text: String) : ContentItem()  // 부드러운 강조
    data class Answer(val text: String) : ContentItem()  // 정답 (문제풀이 내부)
}

sealed class InlinePart {
    data class Text(val text: String) : InlinePart()
    data class Formula(val latex: String) : InlinePart()
}

// 전체 해설 데이터
data class ExplanationData(
    val conceptItems: List<ContentItem> = emptyList(),
    val section1Title: String = "문제 리뷰",
    val section1: List<ContentItem>,
    val section2Title: String = "조건 해석",
    val section2: List<ContentItem>,
    val stepsTitle: String = "문제 풀이",
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
    displayMode: Boolean = false,
    inline: Boolean = false,
    onComplete: () -> Unit = {}
) {
    LatexFormula(
        latex = formula,
        displayMode = displayMode,
        inline = inline,
        onRendered = onComplete
    )
}

@Composable
fun InlineRichLine(
    parts: List<InlinePart>,
    color: Color = Gray600,
    fontSize: androidx.compose.ui.unit.TextUnit = 14.sp,
    fontWeight: FontWeight = FontWeight.Normal,
    lineHeight: androidx.compose.ui.unit.TextUnit = 22.sp,
    onComplete: () -> Unit = {}
) {
    val cssFontSize = fontSize.value.takeIf { it > 0f } ?: 14f
    val cssLineHeight = (lineHeight.value / cssFontSize).takeIf { it > 0f } ?: 1.55f

    LatexParagraph(
        spans = parts.map { part ->
            when (part) {
                is InlinePart.Text -> LatexParagraphSpan.Text(part.text)
                is InlinePart.Formula -> LatexParagraphSpan.Math(part.latex)
            }
        },
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 6.dp),
        textColor = color.toCssRgba(),
        fontSizePx = cssFontSize.toInt(),
        fontWeight = fontWeight.weight,
        lineHeight = cssLineHeight,
        onRendered = onComplete
    )
}

private fun Color.toCssRgba(): String {
    val red = (red * 255).toInt().coerceIn(0, 255)
    val green = (green * 255).toInt().coerceIn(0, 255)
    val blue = (blue * 255).toInt().coerceIn(0, 255)
    return "rgba($red, $green, $blue, $alpha)"
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
            conceptItems = listOf(
                ContentItem.Text("이 문제는 합성함수와 삼각함수가 함께 나온 문제예요."),
                ContentItem.Text("처음에는 식을 억지로 전개하지 말고, 안쪽 식의 구조와 특수값 대입부터 보는 것이 좋아요."),
                ContentItem.Hint("핵심 개념: 합성함수의 미분, 삼각함수의 주기, 극대점 개수 세기")
            ),
            section1Title = "문제 리뷰",
            section1 = listOf(
                ContentItem.Text("안녕하세요! 이 문제를 함께 풀어볼게요. 먼저 문제를 천천히 살펴봅시다."),
                ContentItem.Text("이 문제는 삼각함수가 포함된 합성함수 형태예요. 합성함수란 함수 안에 또 다른 함수가 들어있는 것을 말해요."),
                ContentItem.Text("sin(ax + b + sin x)처럼 sin 안에 또 다른 sin이 들어있죠? 이런 문제는 한 번에 풀려고 하면 어려워요."),
                ContentItem.Hint("핵심 전략: 먼저 특별한 x값(0, 2π 등)을 대입해서 a, b의 값을 찾고, 그 다음에 극대점을 찾는 순서로 풀어요!")
            ),
            section2Title = "문제 해석",
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
            stepsTitle = "문제 풀이",
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
            conceptItems = emptyList(),
            section1Title = "문제 리뷰",
            section1 = listOf(
                ContentItem.Text("이 문제는 처음 보면 삼각함수 안에 또 삼각함수가 들어 있는 합성함수 형태예요. 이런 문제는 식을 무작정 전개하려고 하면 오히려 길이 꼬이기 쉽습니다."),
                ContentItem.Text("그래서 먼저 겉에 있는 sin과 안쪽 식 ax + b + sin x를 분리해서 보는 게 좋아요."),
                ContentItem.Text("또 눈에 띄는 건 0, 2π, 4π 같은 삼각함수의 주기와 딱 맞는 값들이 계속 나온다는 점입니다."),
                ContentItem.Hint("먼저 특수한 x값들을 넣어서 a, b의 후보를 줄이고, 도함수 조건으로 후보를 걸러낸 뒤, 극대점의 개수와 가장 작은 극대점 α₁을 찾는 흐름으로 진행합니다.")
            ),
            section2Title = "문제 해석",
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
            stepsTitle = "문제 풀이",
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
            conceptItems = emptyList(),
            section1Title = "핵심포인트",
            section1 = listOf(
                ContentItem.Text("합성함수 f(x) = sin(ax + b + sin x)에서 특수값 대입으로 a, b를 결정하고, 극대점 개수를 구하는 문제입니다."),
                ContentItem.Hint("접근: (가)로 a, b 후보 → (나)로 필터링 → 극대점 개수 및 α₁ 계산")
            ),
            section2Title = "적용점",
            section2 = listOf(
                ContentItem.Text("(가) 조건에서:"),
                ContentItem.Formula("f(0) = \\sin b = 0 \\;\\Rightarrow\\; b = n\\pi"),
                ContentItem.Formula("f(2\\pi) = 2\\pi a + b \\;\\Rightarrow\\; 2\\pi a + b = 0"),
                ContentItem.Text("따라서 b = −2πa, a = n/2 (n은 정수)"),
                ContentItem.Text("1 ≤ a ≤ 2에서 a ∈ {1, 3/2, 2}"),
                ContentItem.Formula("f'(x) = \\cos(ax + b + \\sin x)(a + \\cos x)"),
                ContentItem.Hint("(나) 조건 검증 시 a = 3/2, b = −3π만 만족")
            ),
            stepsTitle = "실전 적용",
            steps = listOf(
                "핵심 정리" to listOf(
                    ContentItem.Text("g(x) = 3x/2 − 3π + sin x로 치환하면 안쪽 함수가 단조 증가합니다."),
                    ContentItem.Formula("g(0) = -3\\pi, \\; g(4\\pi) = 3\\pi, \\; g'(x) > 0"),
                    ContentItem.Text("따라서 cos(g(x)) = 0인 지점을 세면 극대점 개수가 바로 나옵니다."),
                    ContentItem.Text("극대는 g(x) = −3π/2, π/2, 5π/2에서 생기므로 n = 3입니다."),
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
    imageBase64: String? = null,
    onReset: () -> Unit,
    viewModel: ExplanationViewModel = viewModel(factory = ExplanationViewModel.factory())
) {
    val uiState by viewModel.uiState.collectAsState()
    var currentIndex by remember { mutableIntStateOf(0) }

    LaunchedEffect(imageBase64, explanationLevel) {
        currentIndex = 0
        viewModel.loadExplanation(imageBase64, explanationLevel)
    }

    when (val state = uiState) {
        ExplanationUiState.Idle,
        ExplanationUiState.Loading -> {
            ExplanationLoadingScreen(
                explanationLevel = explanationLevel,
                onReset = onReset
            )
        }

        is ExplanationUiState.Error -> {
            ExplanationErrorScreen(
                message = state.message,
                onBack = onReset,
                onRetry = {
                    currentIndex = 0
                    viewModel.loadExplanation(imageBase64, explanationLevel)
                }
            )
        }

        is ExplanationUiState.Success -> {
            if (explanationLevel == "고급") {
                AdvancedExplanationContentScreen(
                    explanationLevel = explanationLevel,
                    response = state.response,
                    currentIndex = currentIndex,
                    onCurrentIndexChange = { currentIndex = it },
                    onReset = onReset
                )
            } else {
                StandardExplanationContentScreen(
                    explanationLevel = explanationLevel,
                    response = state.response,
                    currentIndex = currentIndex,
                    onCurrentIndexChange = { currentIndex = it },
                    onReset = onReset
                )
            }
        }
    }
}

@Composable
private fun StandardExplanationContentScreen(
    explanationLevel: String,
    response: ExplanationResponse,
    currentIndex: Int,
    onCurrentIndexChange: (Int) -> Unit,
    onReset: () -> Unit
) {
    val explanationData = remember(response) {
        response.toExplanationData(explanationLevel)
    }

    val section1End = explanationData.conceptItems.size + explanationData.section1.size
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

            if (explanationData.conceptItems.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        SectionContent(
                            number = "0",
                            title = "개념설명",
                            items = explanationData.conceptItems,
                            startIndex = 0,
                            currentIndex = currentIndex,
                            onItemComplete = { onCurrentIndexChange(currentIndex + 1) }
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
            }

            // [1. 문제 리뷰 / 핵심포인트] - 흰색 카드
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    SectionContent(
                        number = "1",
                        title = explanationData.section1Title,
                        items = explanationData.section1,
                        startIndex = if (explanationData.conceptItems.isNotEmpty()) {
                            explanationData.conceptItems.size
                        } else {
                            0
                        },
                        currentIndex = currentIndex,
                        onItemComplete = { onCurrentIndexChange(currentIndex + 1) }
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
                            title = explanationData.section2Title,
                            items = explanationData.section2,
                            startIndex = section2Start + 1,
                            currentIndex = currentIndex,
                            onItemComplete = { onCurrentIndexChange(currentIndex + 1) },
                            onHeaderShow = { onCurrentIndexChange(currentIndex + 1) }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // [3. 해설 단계] - 흰색 카드
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
                        if (explanationData.stepsTitle.isNotBlank()) {
                            SectionLabel(number = "3", title = explanationData.stepsTitle)
                            Spacer(modifier = Modifier.height(12.dp))
                        }
                        StepsContent(
                            steps = explanationData.steps,
                            stepIndices = stepIndices,
                            currentIndex = currentIndex,
                            onItemComplete = { onCurrentIndexChange(currentIndex + 1) }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(40.dp))
        }
    }
}

@Composable
private fun AdvancedExplanationContentScreen(
    explanationLevel: String,
    response: ExplanationResponse,
    currentIndex: Int,
    onCurrentIndexChange: (Int) -> Unit,
    onReset: () -> Unit
) {
    val explanationData = remember(response) {
        response.toExplanationData(explanationLevel)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
            .statusBarsPadding()
    ) {
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

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    StaticSubsection(title = explanationData.section1Title, items = explanationData.section1)
                    Spacer(modifier = Modifier.height(16.dp))
                    HorizontalDivider(color = Gray200, thickness = 0.5.dp)
                    Spacer(modifier = Modifier.height(16.dp))
                    StaticSubsection(title = explanationData.section2Title, items = explanationData.section2)
                    Spacer(modifier = Modifier.height(16.dp))
                    HorizontalDivider(color = Gray200, thickness = 0.5.dp)
                    Spacer(modifier = Modifier.height(16.dp))
                    StaticSubsection(
                        title = explanationData.stepsTitle,
                        items = explanationData.steps.firstOrNull()?.second.orEmpty()
                    )
                }
            }

            Spacer(modifier = Modifier.height(40.dp))
        }
    }
}

@Composable
private fun StaticSubsection(
    title: String,
    items: List<ContentItem>
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = title,
            fontSize = 17.sp,
            fontWeight = FontWeight.SemiBold,
            color = Blue900,
            modifier = Modifier.padding(bottom = 10.dp)
        )

        items.forEachIndexed { index, item ->
            if (index > 0) {
                Spacer(modifier = Modifier.height(6.dp))
            }
            when (item) {
                is ContentItem.Text -> {
                    Text(
                        text = item.text,
                        color = Gray600,
                        fontSize = 14.sp,
                        lineHeight = 22.sp
                    )
                }
                is ContentItem.Formula -> {
                    FormulaBox(
                        formula = item.latex,
                        displayMode = item.display
                    )
                }
                is ContentItem.RichLine -> {
                    InlineRichLine(
                        parts = item.parts
                    )
                }
                is ContentItem.Hint -> {
                    HintBox(
                        text = item.text
                    )
                }
                is ContentItem.Answer -> {
                    Text(
                        text = item.text,
                        color = Gray600,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        lineHeight = 22.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun ExplanationLoadingScreen(
    explanationLevel: String,
    onReset: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
            .statusBarsPadding(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
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

        Spacer(modifier = Modifier.weight(1f))
        CircularProgressIndicator(color = Blue500)
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "AI 해설을 생성하고 있습니다",
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
            color = Blue900
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "OCR, 해설 후보 생성, 검증 과정을 진행 중입니다",
            fontSize = 13.sp,
            color = Gray500,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )
        Spacer(modifier = Modifier.weight(1f))
    }
}

@Composable
private fun ExplanationErrorScreen(
    message: String,
    onBack: () -> Unit,
    onRetry: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
            .statusBarsPadding()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "해설 생성 실패",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = Blue900
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = message,
            fontSize = 14.sp,
            color = Gray600,
            lineHeight = 20.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onRetry,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Blue500)
        ) {
            Text("다시 시도")
        }
        TextButton(onClick = onBack) {
            Text("홈으로 돌아가기", color = Gray600)
        }
    }
}

private fun ExplanationResponse.toExplanationData(level: String): ExplanationData {
    val parsedConceptItems = conceptExplanation.toContentItems()
    val parsedReviewItems = problemReview.toContentItems()
    val parsedConditionItems = conditionInterpretation.toContentItems()
    val parsedSolutionItems = solution.toContentItems()

    val reviewItems = parsedReviewItems
        .ifEmpty { listOf(ContentItem.Text("문제 리뷰가 비어 있습니다.")) }
    val conditionItems = parsedConditionItems
        .ifEmpty { listOf(ContentItem.Text("조건 해석이 비어 있습니다.")) }
    val solutionItems = parsedSolutionItems.toMutableList()

    if (answer.isNotBlank() && solutionItems.none { it is ContentItem.Answer }) {
        solutionItems.add(ContentItem.Answer("답: $answer"))
    }

    val keyPointItems = parsedReviewItems.ifEmpty { keyPoints.toContentItems() }.ifEmpty { reviewItems }
    val approachItems = parsedConditionItems.ifEmpty { approachPerspectives.toContentItems() }.ifEmpty { conditionItems }
    val transferableItems = solutionItems.ifEmpty { transferableInsight.toContentItems() }

    val baseSolutionItems = solutionItems.ifEmpty {
        listOf(ContentItem.Text("문제 풀이가 비어 있습니다."))
    }

    return when (level) {
        "초급" -> ExplanationData(
            conceptItems = parsedConceptItems,
            section1Title = "문제 리뷰",
            section1 = reviewItems,
            section2Title = "문제 해석",
            section2 = conditionItems,
            stepsTitle = "문제 풀이",
            steps = listOf("도함수와 극대점" to baseSolutionItems)
        )
        "고급" -> ExplanationData(
            conceptItems = emptyList(),
            section1Title = "핵심 포인트",
            section1 = keyPointItems,
            section2Title = "적용점",
            section2 = approachItems,
            stepsTitle = "최종 정리",
            steps = listOf("최종 정리" to transferableItems)
        )
        else -> ExplanationData(
            conceptItems = emptyList(),
            section1Title = "문제 리뷰",
            section1 = reviewItems,
            section2Title = "문제 해석",
            section2 = conditionItems,
            stepsTitle = "문제 풀이",
            steps = listOf("문제 풀이" to baseSolutionItems)
        )
    }
}

private fun List<ApiExplanationBlock>.toContentItems(): List<ContentItem> {
    return flatMap { block ->
        when (block) {
            is ApiExplanationBlock.Paragraph -> {
                val parts = block.content.mapNotNull { span ->
                    when (span) {
                        is ApiInlineSpan.Text -> {
                            if (span.text.isBlank()) null else InlinePart.Text(span.text)
                        }
                        is ApiInlineSpan.Latex -> {
                            if (span.text.isBlank()) null else InlinePart.Formula(span.text)
                        }
                    }
                }

                if (parts.isEmpty()) {
                    emptyList()
                } else if (parts.size == 1 && parts[0] is InlinePart.Text) {
                    listOf(ContentItem.Text((parts[0] as InlinePart.Text).text))
                } else {
                    listOf(ContentItem.RichLine(parts))
                }
            }
            is ApiExplanationBlock.Math -> {
                if (block.content.isBlank()) {
                    emptyList()
                } else {
                    listOf(ContentItem.Formula(block.content, display = true))
                }
            }
        }
    }
}

private fun String.toContentItems(): List<ContentItem> {
    val normalized = replace("\\n", "\n").trim()
    if (normalized.isBlank()) return emptyList()

    val parsedItems = mutableListOf<ContentItem>()
    normalized.lines()
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .forEach { line ->
            val answerLine = line.startsWith("답:")
            when {
                answerLine -> parsedItems.add(ContentItem.Answer(line))
                else -> {
                    parsedItems.add(parseLineContent(line))
                }
            }
        }

    val mergedItems = mutableListOf<ContentItem>()
    var pendingInlineParts: MutableList<InlinePart>? = null

    fun flushPendingIfNeeded() {
        val pending = pendingInlineParts ?: return
        if (pending.isNotEmpty()) {
            mergedItems.add(ContentItem.RichLine(pending.toList()))
        }
        pendingInlineParts = null
    }

    fun isShortInlineFormulaLine(item: ContentItem): Boolean {
        return item is ContentItem.RichLine &&
            item.parts.size == 1 &&
            item.parts[0] is InlinePart.Formula
    }

    fun appendInlinePartToLast(inlinePart: InlinePart) {
        val previous = mergedItems.lastOrNull()
        when (previous) {
            is ContentItem.Text -> {
                mergedItems[mergedItems.lastIndex] = ContentItem.RichLine(
                    listOf(
                        InlinePart.Text(previous.text),
                        InlinePart.Text(" "),
                        inlinePart
                    )
                )
            }

            is ContentItem.RichLine -> {
                mergedItems[mergedItems.lastIndex] = previous.copy(
                    parts = previous.parts + InlinePart.Text(" ") + inlinePart
                )
            }

            else -> {
                val pending = pendingInlineParts ?: mutableListOf()
                pending.add(inlinePart)
                pendingInlineParts = pending
            }
        }
    }

    parsedItems.forEach { item ->
        when (item) {
            is ContentItem.Answer -> {
                flushPendingIfNeeded()
                mergedItems.add(item)
            }

            is ContentItem.Formula -> {
                flushPendingIfNeeded()
                mergedItems.add(item)
            }

            is ContentItem.Text -> {
                if (pendingInlineParts != null && pendingInlineParts!!.isNotEmpty()) {
                    val pending = pendingInlineParts!!.toList()
                    pendingInlineParts = null
                    mergedItems.add(
                        ContentItem.RichLine(
                            pending + InlinePart.Text(" ") + InlinePart.Text(item.text)
                        )
                    )
                } else {
                    mergedItems.add(item)
                }
            }

            is ContentItem.Hint -> {
                flushPendingIfNeeded()
                mergedItems.add(item)
            }

            is ContentItem.RichLine -> {
                if (item.parts.size == 1 && item.parts[0] is InlinePart.Formula) {
                    val formula = (item.parts[0] as InlinePart.Formula).latex
                    if (shouldDisplayFormula(formula, formula)) {
                        flushPendingIfNeeded()
                        mergedItems.add(ContentItem.Formula(formula, display = true))
                    } else {
                        val pending = pendingInlineParts
                        if (pending != null && pending.isNotEmpty()) {
                            mergedItems.add(
                                ContentItem.RichLine(
                                    pending + InlinePart.Text(" ") + item.parts
                                )
                            )
                            pendingInlineParts = null
                        } else {
                            appendInlinePartToLast(item.parts[0])
                        }
                    }
                } else {
                    if (pendingInlineParts != null && pendingInlineParts!!.isNotEmpty()) {
                        val pending = pendingInlineParts!!.toList()
                        pendingInlineParts = null
                        mergedItems.add(
                            ContentItem.RichLine(
                                pending + InlinePart.Text(" ") + item.parts
                            )
                        )
                    } else {
                        mergedItems.add(item)
                    }
                }
            }
        }
    }

    flushPendingIfNeeded()
    return mergedItems
}

private fun parseLineContent(line: String): ContentItem {
    val trimmed = line.trim()
    if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
        val blockFormula = trimmed.removePrefix("$$").removeSuffix("$$").trim()
        if (blockFormula.isNotBlank()) {
            return ContentItem.Formula(blockFormula, display = true)
        }
    }

    if (trimmed.startsWith("\\[") && trimmed.endsWith("\\]") && trimmed.length > 4) {
        val blockFormula = trimmed.removePrefix("\\[").removeSuffix("\\]").trim()
        if (blockFormula.isNotBlank()) {
            return ContentItem.Formula(blockFormula, display = true)
        }
    }

    if (trimmed.startsWith("\\(") && trimmed.endsWith("\\)") && trimmed.length > 4) {
        val inlineFormula = trimmed.removePrefix("\\(").removeSuffix("\\)").trim()
        if (inlineFormula.isNotBlank()) {
            return ContentItem.RichLine(listOf(InlinePart.Formula(inlineFormula)))
        }
    }

    val regex = Regex("""\$\$(.+?)\$\$|\$(.+?)\$""")
    val parts = mutableListOf<InlinePart>()
    var lastIndex = 0

    regex.findAll(line).forEach { match ->
        val textBefore = line.substring(lastIndex, match.range.first)
        if (textBefore.isNotBlank()) {
            parts.add(InlinePart.Text(textBefore))
        }

        val formula = (match.groupValues[1].ifBlank { match.groupValues[2] }).trim()
        if (formula.isNotBlank()) {
            if (shouldDisplayFormula(formula, line) && parts.isEmpty() && line.trim() == match.value.trim()) {
                return ContentItem.Formula(formula, display = true)
            }
            parts.add(InlinePart.Formula(formula))
        }
        lastIndex = match.range.last + 1
    }

    val remainingText = line.substring(lastIndex)
    if (remainingText.isNotBlank()) {
        parts.add(InlinePart.Text(remainingText))
    }

    if (parts.isEmpty()) {
        return ContentItem.Text(line)
    }

    if (parts.size == 1 && parts[0] is InlinePart.Formula) {
        val formula = (parts[0] as InlinePart.Formula).latex
        if (shouldDisplayFormula(formula, line)) {
            return ContentItem.Formula(formula, display = true)
        }
    }

    if (parts.size == 1 && parts[0] is InlinePart.Text) {
        return ContentItem.Text((parts[0] as InlinePart.Text).text.trim())
    }

    return ContentItem.RichLine(parts)
}

private fun shouldDisplayFormula(formula: String, line: String): Boolean {
    val trimmedFormula = formula.trim()
    val displayMarkers = listOf("\\frac", "\\sum", "\\int", "\\lim", "\\prod", "\\begin", "\\displaystyle")
    if (displayMarkers.any { trimmedFormula.contains(it) }) {
        return true
    }

    if (trimmedFormula.length >= 28) {
        return true
    }

    val relationalMarkers = listOf("=", "\\Rightarrow", "\\Leftrightarrow")
    if (relationalMarkers.any { trimmedFormula.contains(it) } && trimmedFormula.length >= 20) {
        return true
    }

    if (trimmedFormula.count { it == '=' } >= 2) {
        return true
    }

    return false
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
                                displayMode = item.display,
                                onComplete = onItemComplete
                            )
                        }
                        is ContentItem.RichLine -> {
                            InlineRichLine(
                                parts = item.parts,
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
                                        displayMode = item.display,
                                        onComplete = onItemComplete
                                    )
                                }
                                is ContentItem.RichLine -> {
                                    InlineRichLine(
                                        parts = item.parts,
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
