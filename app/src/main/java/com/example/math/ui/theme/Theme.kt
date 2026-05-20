package com.example.math.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val LightColorScheme = lightColorScheme(
    primary = Blue500,
    onPrimary = androidx.compose.ui.graphics.Color.White,
    primaryContainer = Blue100,
    onPrimaryContainer = Blue900,
    secondary = Purple600,
    onSecondary = androidx.compose.ui.graphics.Color.White,
    secondaryContainer = Purple50,
    onSecondaryContainer = TextPrimary,
    background = Blue50,
    onBackground = TextPrimary,
    surface = androidx.compose.ui.graphics.Color.White,
    onSurface = TextPrimary,
    surfaceVariant = Gray100,
    onSurfaceVariant = TextSecondary,
    outline = Gray300
)

private val DarkColorScheme = darkColorScheme(
    primary = Blue500,
    onPrimary = androidx.compose.ui.graphics.Color.White,
    primaryContainer = Blue600,
    onPrimaryContainer = Blue100,
    secondary = Purple600,
    onSecondary = androidx.compose.ui.graphics.Color.White,
    background = androidx.compose.ui.graphics.Color(0xFF0F172A),
    onBackground = androidx.compose.ui.graphics.Color.White,
    surface = androidx.compose.ui.graphics.Color(0xFF1E293B),
    onSurface = androidx.compose.ui.graphics.Color.White
)

@Composable
fun MathAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}