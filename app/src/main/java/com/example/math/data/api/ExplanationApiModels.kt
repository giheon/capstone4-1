package com.example.math.data.api

import org.json.JSONArray
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

sealed class ExplanationBlock {
    data class Paragraph(val content: List<InlineSpan>) : ExplanationBlock()
    data class Math(val content: String) : ExplanationBlock()
}

sealed class InlineSpan {
    data class Text(val text: String) : InlineSpan()
    data class Latex(val text: String) : InlineSpan()
}

data class ExplanationResponse(
    val problemText: String,
    val questionType: String,
    val subject: String,
    val difficulty: String,
    val unit: String,
    val selectedModel: String,
    val problemReview: List<ExplanationBlock>,
    val conditionInterpretation: List<ExplanationBlock>,
    val solution: List<ExplanationBlock>,
    val keyPoints: String,
    val approachPerspectives: String,
    val transferableInsight: String,
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
                problemReview = parseBlocks(json, "problem_review"),
                conditionInterpretation = parseBlocks(json, "condition_interpretation"),
                solution = parseBlocks(json, "solution"),
                keyPoints = json.optString("key_points"),
                approachPerspectives = json.optString("approach_perspectives"),
                transferableInsight = json.optString("transferable_insight"),
                answer = json.optString("answer"),
                majorityAnswer = json.optString("majority_answer"),
                isComplete = json.optBoolean("is_complete")
            )
        }

        private fun parseBlocks(json: JSONObject, key: String): List<ExplanationBlock> {
            val blocksArray = json.optJSONArray(key)
            if (blocksArray != null) {
                return parseBlocks(blocksArray)
            }

            val legacyText = json.optString(key)
            if (legacyText.isBlank()) {
                return emptyList()
            }

            return legacyText
                .replace("\\n", "\n")
                .lines()
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .flatMap { parseLegacyLine(it) }
        }

        private fun parseBlocks(array: JSONArray): List<ExplanationBlock> {
            return (0 until array.length()).mapNotNull { index ->
                val block = array.optJSONObject(index) ?: return@mapNotNull null
                when (block.optString("type")) {
                    "paragraph" -> {
                        val spans = parseSpans(block.optJSONArray("content"))
                        if (spans.isEmpty()) null else ExplanationBlock.Paragraph(spans)
                    }
                    "math" -> {
                        val content = block.optString("content").trim()
                        if (content.isBlank()) null else ExplanationBlock.Math(stripMathDelimiters(content))
                    }
                    "text" -> {
                        val text = block.optString("text", block.optString("content"))
                        if (text.isBlank()) null else ExplanationBlock.Paragraph(listOf(InlineSpan.Text(text)))
                    }
                    "latex" -> {
                        val latex = block.optString("text", block.optString("content")).trim()
                        if (latex.isBlank()) null else ExplanationBlock.Math(stripMathDelimiters(latex))
                    }
                    else -> null
                }
            }
        }

        private fun parseSpans(array: JSONArray?): List<InlineSpan> {
            if (array == null) {
                return emptyList()
            }

            return (0 until array.length()).mapNotNull { index ->
                val span = array.optJSONObject(index) ?: return@mapNotNull null
                val rawText = span.optString("text", span.optString("content"))
                if (rawText.isBlank()) {
                    return@mapNotNull null
                }

                when (span.optString("type")) {
                    "latex" -> InlineSpan.Latex(stripMathDelimiters(rawText.trim()))
                    "text" -> InlineSpan.Text(rawText)
                    else -> null
                }
            }
        }

        private fun parseLegacyLine(line: String): List<ExplanationBlock> {
            val trimmed = line.trim()
            val displayPairs = listOf(
                "$$" to "$$",
                "\\[" to "\\]"
            )

            displayPairs.forEach { (prefix, suffix) ->
                if (trimmed.startsWith(prefix) && trimmed.endsWith(suffix) && trimmed.length > prefix.length + suffix.length) {
                    return listOf(ExplanationBlock.Math(stripMathDelimiters(trimmed)))
                }
            }

            val regex = Regex("""\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]""")
            val spans = mutableListOf<InlineSpan>()
            var lastIndex = 0

            regex.findAll(line).forEach { match ->
                val textBefore = line.substring(lastIndex, match.range.first)
                if (textBefore.isNotEmpty()) {
                    spans.add(InlineSpan.Text(textBefore))
                }

                val latex = match.groupValues
                    .drop(1)
                    .firstOrNull { it.isNotBlank() }
                    ?.trim()
                    .orEmpty()

                if (latex.isNotBlank()) {
                    spans.add(InlineSpan.Latex(stripMathDelimiters(latex)))
                }
                lastIndex = match.range.last + 1
            }

            val remainingText = line.substring(lastIndex)
            if (remainingText.isNotEmpty()) {
                spans.add(InlineSpan.Text(remainingText))
            }

            if (spans.isEmpty()) {
                spans.add(InlineSpan.Text(line))
            }

            return listOf(ExplanationBlock.Paragraph(spans))
        }

        private fun stripMathDelimiters(raw: String): String {
            var text = raw.trim()
            val pairs = listOf(
                "$$" to "$$",
                "$" to "$",
                "\\(" to "\\)",
                "\\[" to "\\]"
            )

            var changed = true
            while (changed) {
                changed = false
                pairs.forEach { (prefix, suffix) ->
                    if (text.startsWith(prefix) && text.endsWith(suffix) && text.length > prefix.length + suffix.length) {
                        text = text.removePrefix(prefix).removeSuffix(suffix).trim()
                        changed = true
                    }
                }
            }

            return text
        }
    }
}
