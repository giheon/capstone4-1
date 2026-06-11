package com.example.math.data.api

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit
import android.util.Log

class MathExplanationApiClient(
    baseUrl: String,
    private val client: OkHttpClient = defaultClient()
) {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    suspend fun generateExplanation(request: ExplanationRequest): ExplanationResponse {
        return withContext(Dispatchers.IO) {
            val httpRequest = Request.Builder()
                .url("$normalizedBaseUrl/explain")
                .post(request.toJson().toString().toRequestBody(jsonMediaType))
                .build()

            client.newCall(httpRequest).execute().use { response ->
                val body = response.body?.string().orEmpty()
                Log.d("ExplainApiRaw", body)
                if (!response.isSuccessful) {
                    val message = parseErrorMessage(body)
                    throw IOException("API ${response.code}: $message")
                }
                ExplanationResponse.fromJson(JSONObject(body))
            }
        }
    }

    private fun parseErrorMessage(body: String): String {
        if (body.isBlank()) return "empty response"
        return runCatching {
            JSONObject(body).optString("detail", body)
        }.getOrDefault(body)
    }

    companion object {
        private fun defaultClient(): OkHttpClient {
            return OkHttpClient.Builder()
                .connectTimeout(20, TimeUnit.SECONDS)
                .readTimeout(420, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build()
        }
    }
}
