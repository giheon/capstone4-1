package com.example.math.ui.screens.levelselect

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
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
import com.example.math.ui.theme.*

data class ExplanationLevel(
    val id: String,
    val title: String,
    val subtitle: String,
    val description: String
)

@Composable
fun LevelSelectScreen(
    onBack: () -> Unit,
    onLevelSelected: (String) -> Unit
) {
    var selectedLevel by remember { mutableStateOf<String?>(null) }

    val levels = remember {
        listOf(
            ExplanationLevel(
                id = "초급",
                title = "초급",
                subtitle = "기초부터 차근차근",
                description = "수학 개념이 익숙하지 않은 학생을 위한 상세한 설명"
            ),
            ExplanationLevel(
                id = "중급",
                title = "중급",
                subtitle = "핵심 포인트 중심",
                description = "기본 개념을 알고 있는 학생을 위한 효율적인 풀이"
            ),
            ExplanationLevel(
                id = "고급",
                title = "고급",
                subtitle = "빠르고 간결하게",
                description = "실력자를 위한 핵심만 담은 간결한 해설"
            )
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // 헤더
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                FilledIconButton(
                    onClick = onBack,
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
            }

            // 타이틀
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
                    .padding(top = 16.dp, bottom = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "해설 수준 선택",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Blue900
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "원하는 해설의 상세 수준을 선택해주세요",
                    fontSize = 14.sp,
                    color = Gray500,
                    textAlign = TextAlign.Center
                )
            }

            // 레벨 카드들
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                levels.forEach { level ->
                    LevelCard(
                        level = level,
                        isSelected = selectedLevel == level.id,
                        onClick = { selectedLevel = level.id }
                    )
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            // 확인 버튼
            Button(
                onClick = { selectedLevel?.let { onLevelSelected(it) } },
                enabled = selectedLevel != null,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 40.dp)
                    .height(56.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Blue500,
                    disabledContainerColor = Gray300
                )
            ) {
                Text(
                    text = "해설 보기",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
private fun LevelCard(
    level: ExplanationLevel,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val backgroundColor by animateColorAsState(
        targetValue = if (isSelected) Blue50 else Color.White,
        animationSpec = tween(200),
        label = "bgColor"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) Blue500 else Gray200,
        animationSpec = tween(200),
        label = "borderColor"
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .border(
                width = if (isSelected) 2.dp else 1.dp,
                color = borderColor,
                shape = RoundedCornerShape(16.dp)
            ),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = backgroundColor),
        elevation = CardDefaults.cardElevation(
            defaultElevation = if (isSelected) 4.dp else 1.dp
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 체크 아이콘
            Box(
                modifier = Modifier
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(if (isSelected) Blue500 else Gray200),
                contentAlignment = Alignment.Center
            ) {
                if (isSelected) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.width(16.dp))

            // 텍스트 내용
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = level.title,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isSelected) Blue600 else Blue900
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = level.subtitle,
                        fontSize = 13.sp,
                        color = Gray500
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = level.description,
                    fontSize = 13.sp,
                    color = Gray500,
                    lineHeight = 18.sp
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun LevelSelectScreenPreview() {
    MathAITheme {
        LevelSelectScreen(
            onBack = {},
            onLevelSelected = {}
        )
    }
}
