package com.example.math.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.example.math.data.CapturedImageHolder
import com.example.math.ui.screens.camera.CameraScanScreen
import com.example.math.ui.screens.explanation.ExplanationScreen
import com.example.math.ui.screens.home.HomeScreen
import com.example.math.ui.screens.levelselect.LevelSelectScreen

sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Camera : Screen("camera")
    object LevelSelect : Screen("level_select")
    object Explanation : Screen("explanation/{level}") {
        fun createRoute(level: String) = "explanation/$level"
    }
}

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Screen.Home.route
    ) {
        composable(Screen.Home.route) {
            HomeScreen(
                onStartCamera = { navController.navigate(Screen.Camera.route) }
            )
        }

        composable(Screen.Camera.route) {
            CameraScanScreen(
                onBack = { navController.popBackStack() },
                onCapture = { base64Image ->
                    // 촬영된 이미지를 저장하고 수준 선택 화면으로 이동
                    CapturedImageHolder.setImage(base64Image)
                    navController.navigate(Screen.LevelSelect.route)
                }
            )
        }

        composable(Screen.LevelSelect.route) {
            LevelSelectScreen(
                onBack = { navController.popBackStack() },
                onLevelSelected = { level ->
                    // 선택한 수준과 함께 해설 화면으로 이동
                    navController.navigate(Screen.Explanation.createRoute(level))
                }
            )
        }

        composable(
            route = Screen.Explanation.route,
            arguments = listOf(
                navArgument("level") {
                    type = NavType.StringType
                    defaultValue = "중급"
                }
            )
        ) { backStackEntry ->
            val level = backStackEntry.arguments?.getString("level") ?: "중급"
            ExplanationScreen(
                explanationLevel = level,
                imageBase64 = CapturedImageHolder.imageBase64,
                onReset = {
                    // 이미지 초기화 후 홈으로 이동
                    CapturedImageHolder.clear()
                    navController.popBackStack(Screen.Home.route, inclusive = false)
                }
            )
        }
    }
}
