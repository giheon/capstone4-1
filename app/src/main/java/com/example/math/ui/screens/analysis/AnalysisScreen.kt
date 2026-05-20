package com.example.math.ui.screens.analysis

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.math.ui.theme.*

data class SubjectScore(
    val name: String,
    val score: Int,
    val color: Color
)

data class WeeklyData(
    val day: String,
    val problems: Int
)

@Composable
fun AnalysisScreen(onBack: () -> Unit) {
    val subjectData = remember {
        listOf(
            SubjectScore("미적분", 85, Blue500),
            SubjectScore("확률통계", 72, Purple600),
            SubjectScore("기하", 65, Pink600)
        )
    }

    val weekData = remember {
        listOf(
            WeeklyData("월", 8),
            WeeklyData("화", 12),
            WeeklyData("수", 6),
            WeeklyData("목", 15),
            WeeklyData("금", 11),
            WeeklyData("토", 9),
            WeeklyData("일", 5)
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        // Top bar
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = Color.White,
            shadowElevation = 2.dp
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .background(Gray50)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = ripple(bounded = true)
                        ) { onBack() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.ArrowBack,
                        contentDescription = "뒤로가기",
                        tint = Gray600,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "성적 분석",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    color = TextPrimary
                )
            }
        }

        // Content
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            // Overall Score Card
            OverallScoreCard()

            Spacer(modifier = Modifier.height(16.dp))

            // Subject Scores
            SubjectScoresSection(subjectData)

            Spacer(modifier = Modifier.height(16.dp))

            // Weekly Chart
            WeeklyChartSection(weekData)

            Spacer(modifier = Modifier.height(16.dp))

            // Improvement Goals
            ImprovementGoalsSection()
        }
    }
}

@Composable
private fun OverallScoreCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Transparent
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    brush = Brush.linearGradient(
                        colors = listOf(Purple50, Pink50)
                    )
                )
                .padding(16.dp)
        ) {
            Column {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.EmojiEvents,
                        contentDescription = null,
                        tint = Purple600,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "종합 성적",
                        fontSize = 14.sp,
                        color = TextPrimary
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "74.3",
                        fontSize = 36.sp,
                        fontWeight = FontWeight.Bold,
                        color = Purple600
                    )
                    Text(
                        text = "평균 점수",
                        fontSize = 12.sp,
                        color = Gray600
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.TrendingUp,
                            contentDescription = null,
                            tint = Green500,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "지난주 대비 +5.2점",
                            fontSize = 12.sp,
                            color = Green600
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SubjectScoresSection(subjects: List<SubjectScore>) {
    Column {
        Text(
            text = "과목별 성적",
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                subjects.forEachIndexed { index, subject ->
                    SubjectProgressBar(subject)
                    if (index < subjects.lastIndex) {
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun SubjectProgressBar(subject: SubjectScore) {
    var animatedProgress by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(subject.score) {
        animate(
            initialValue = 0f,
            targetValue = subject.score / 100f,
            animationSpec = tween(600, delayMillis = 200)
        ) { value, _ ->
            animatedProgress = value
        }
    }

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = subject.name,
                fontSize = 14.sp,
                color = TextPrimary
            )
            Text(
                text = "${subject.score}점",
                fontSize = 14.sp,
                color = subject.color
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Gray100)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(animatedProgress)
                    .fillMaxHeight()
                    .clip(RoundedCornerShape(4.dp))
                    .background(subject.color)
            )
        }
    }
}

@Composable
private fun WeeklyChartSection(weekData: List<WeeklyData>) {
    val maxProblems = weekData.maxOf { it.problems }

    Column {
        Text(
            text = "주간 학습량",
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.Bottom
            ) {
                weekData.forEach { data ->
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Box(
                            modifier = Modifier
                                .width(24.dp)
                                .height((data.problems.toFloat() / maxProblems * 100).dp)
                                .clip(RoundedCornerShape(topStart = 6.dp, topEnd = 6.dp))
                                .background(Blue500)
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = data.day,
                            fontSize = 11.sp,
                            color = Gray500
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ImprovementGoalsSection() {
    Column {
        Text(
            text = "개선 목표",
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(12.dp))

        GoalCard(
            icon = Icons.Default.GpsFixed,
            iconColor = Blue500,
            title = "기하 영역 집중 학습",
            description = "공간벡터와 이차곡선 문제를 매일 3개씩 풀어보세요"
        )

        Spacer(modifier = Modifier.height(8.dp))

        GoalCard(
            icon = Icons.Default.GpsFixed,
            iconColor = Purple600,
            title = "풀이 시간 단축",
            description = "평균 풀이 시간을 3분 이내로 줄여보세요"
        )
    }
}

@Composable
private fun GoalCard(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    iconColor: Color,
    title: String,
    description: String
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = iconColor,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Column {
                Text(
                    text = title,
                    fontSize = 14.sp,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = description,
                    fontSize = 12.sp,
                    color = Gray600
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun AnalysisScreenPreview() {
    MathAITheme {
        AnalysisScreen(onBack = {})
    }
}