package com.example.math.ui.screens.explanation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.math.BuildConfig
import com.example.math.data.api.ExplanationResponse
import com.example.math.data.api.MathExplanationApiClient
import com.example.math.data.repository.ExplanationRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface ExplanationUiState {
    data object Idle : ExplanationUiState
    data object Loading : ExplanationUiState
    data class Success(val response: ExplanationResponse) : ExplanationUiState
    data class Error(val message: String) : ExplanationUiState
}

class ExplanationViewModel(
    private val repository: ExplanationRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow<ExplanationUiState>(ExplanationUiState.Idle)
    val uiState: StateFlow<ExplanationUiState> = _uiState.asStateFlow()

    fun loadExplanation(imageBase64: String?, explanationLevel: String) {
        if (imageBase64.isNullOrBlank()) {
            _uiState.value = ExplanationUiState.Error("촬영된 이미지가 없습니다. 다시 촬영해주세요.")
            return
        }

        _uiState.value = ExplanationUiState.Loading
        viewModelScope.launch {
            _uiState.value = runCatching {
                repository.generateExplanation(
                    imageBase64 = imageBase64,
                    explanationLevel = explanationLevel
                )
            }.fold(
                onSuccess = { ExplanationUiState.Success(it) },
                onFailure = { error ->
                    ExplanationUiState.Error(error.message ?: "해설 생성 중 오류가 발생했습니다.")
                }
            )
        }
    }

    companion object {
        fun factory(): ViewModelProvider.Factory {
            return object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    val apiClient = MathExplanationApiClient(BuildConfig.API_BASE_URL)
                    val repository = ExplanationRepository(apiClient)
                    return ExplanationViewModel(repository) as T
                }
            }
        }
    }
}
