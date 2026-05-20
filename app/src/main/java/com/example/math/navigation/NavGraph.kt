package com.example.math.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.math.ui.screens.camera.CameraScanScreen
import com.example.math.ui.screens.explanation.ExplanationScreen
import com.example.math.ui.screens.home.HomeScreen

sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Camera : Screen("camera")
    object Explanation : Screen("explanation")
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
                onCapture = { navController.navigate(Screen.Explanation.route) }
            )
        }

        composable(Screen.Explanation.route) {
            ExplanationScreen(
                onReset = {
                    navController.popBackStack(Screen.Home.route, inclusive = false)
                }
            )
        }
    }
}
