package com.example.math.data.repository

import com.example.math.data.api.ExplanationRequest
import com.example.math.data.api.ExplanationResponse
import com.example.math.data.api.MathExplanationApiClient
import java.util.UUID

class ExplanationRepository(
    private val apiClient: MathExplanationApiClient
) {
    suspend fun generateExplanation(
        imageBase64: String,
        explanationLevel: String
    ): ExplanationResponse {
        return apiClient.generateExplanation(
            ExplanationRequest(
                imageBase64 = imageBase64,
                explanationLevel = explanationLevel,
                clientSessionId = UUID.randomUUID().toString(),
                clientMetadata = mapOf(
                    "app_feature" to "camera_explanation",
                    "request_format" to "base64"
                )
            )
        )
    }
}
