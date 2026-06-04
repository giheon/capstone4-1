package com.example.math.data.api

import org.json.JSONObject

data class ExplanationRequest(
    val imageBase64: String,
    val explanationLevel: String,
    val clientSessionId: String,
    val clientSource: String = "android",
    val clientMetadata: Map<String, String> = emptyMap()
) {
    fun toJson(): JSONObject {
        val metadataJson = JSONObject()
        clientMetadata.forEach { (key, value) ->
            metadataJson.put(key, value)
        }

        return JSONObject()
            .put("image_base64", imageBase64)
            .put("explanation_level", explanationLevel)
            .put("client_session_id", clientSessionId)
            .put("client_source", clientSource)
            .put("client_metadata", metadataJson)
    }
}

data class ExplanationResponse(
    val problemText: String,
    val questionType: String,
    val subject: String,
    val difficulty: String,
    val unit: String,
    val selectedModel: String,
    val problemReview: String,
    val conditionInterpretation: String,
    val solution: String,
    val answer: String,
    val majorityAnswer: String,
    val isComplete: Boolean
) {
    companion object {
        fun fromJson(json: JSONObject): ExplanationResponse {
            return ExplanationResponse(
                problemText = json.optString("problem_text"),
                questionType = json.optString("question_type"),
                subject = json.optString("subject"),
                difficulty = json.optString("difficulty"),
                unit = json.optString("unit"),
                selectedModel = json.optString("selected_model"),
                problemReview = json.optString("problem_review"),
                conditionInterpretation = json.optString("condition_interpretation"),
                solution = json.optString("solution"),
                answer = json.optString("answer"),
                majorityAnswer = json.optString("majority_answer"),
                isComplete = json.optBoolean("is_complete")
            )
        }
    }
}
