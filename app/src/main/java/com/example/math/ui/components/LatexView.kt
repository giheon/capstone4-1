package com.example.math.ui.components

import android.annotation.SuppressLint
import android.graphics.Color as AndroidColor
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.math.ui.theme.Gray100
import java.net.URLEncoder
import kotlin.math.max

sealed class LatexParagraphSpan {
    data class Text(val text: String) : LatexParagraphSpan()
    data class Math(val latex: String) : LatexParagraphSpan()
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun LatexFormula(
    latex: String,
    displayMode: Boolean = false,
    inline: Boolean = false,
    modifier: Modifier = Modifier,
    onRendered: () -> Unit = {}
) {
    var hasRendered by remember { mutableStateOf(false) }
    val encodedLatex = remember(latex) {
        URLEncoder.encode(latex, "UTF-8")
    }
    val encodedDisplayMode = if (displayMode) "1" else "0"

    val containerModifier = if (inline) {
        modifier
            .wrapContentWidth()
            .wrapContentHeight()
            .padding(horizontal = 0.dp, vertical = 0.dp)
    } else {
        modifier
            .fillMaxWidth()
            .padding(vertical = if (displayMode) 6.dp else 2.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Gray100)
    }

    Box(
        modifier = containerModifier,
        contentAlignment = if (inline) Alignment.CenterStart else Alignment.Center
    ) {
        AndroidView(
            modifier = Modifier
                .then(if (inline) Modifier.wrapContentWidth() else Modifier.fillMaxWidth())
                .wrapContentHeight()
                .defaultMinSize(minHeight = when {
                    displayMode -> 42.dp
                    inline -> 0.dp
                    else -> 26.dp
                }),
            factory = { context ->
                WebView(context).apply {
                    settings.apply {
                        javaScriptEnabled = true
                        loadWithOverviewMode = true
                        useWideViewPort = true
                        cacheMode = WebSettings.LOAD_DEFAULT
                        domStorageEnabled = true
                    }
                    setBackgroundColor(AndroidColor.TRANSPARENT)
                    setLayerType(android.view.View.LAYER_TYPE_HARDWARE, null)

                    webViewClient = object : WebViewClient() {
                        override fun onPageFinished(view: WebView?, url: String?) {
                            super.onPageFinished(view, url)
                            if (!hasRendered) {
                                hasRendered = true
                                onRendered()
                            }
                        }
                    }

                    loadUrl("file:///android_asset/katex.html?latex=$encodedLatex&display=$encodedDisplayMode")
                }
            },
            update = { webView ->
                webView.loadUrl("file:///android_asset/katex.html?latex=$encodedLatex&display=$encodedDisplayMode")
            }
        )
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun LatexParagraph(
    spans: List<LatexParagraphSpan>,
    modifier: Modifier = Modifier,
    textColor: String = "#4B5563",
    fontSizePx: Int = 14,
    fontWeight: Int = 400,
    lineHeight: Float = 1.55f,
    minHeight: Dp = 24.dp,
    onRendered: () -> Unit = {}
) {
    val density = LocalDensity.current
    val html = remember(spans, textColor, fontSizePx, fontWeight, lineHeight) {
        buildLatexParagraphHtml(
            spans = spans,
            textColor = textColor,
            fontSizePx = fontSizePx,
            fontWeight = fontWeight,
            lineHeight = lineHeight
        )
    }
    var hasRendered by remember(html) { mutableStateOf(false) }
    var height by remember(html) { mutableStateOf(minHeight) }

    val bridge = remember(html) {
        LatexParagraphBridge { heightPx ->
            height = with(density) {
                max(heightPx, 1).toDp()
            }.coerceAtLeast(minHeight)
            if (!hasRendered) {
                hasRendered = true
                onRendered()
            }
        }
    }

    AndroidView(
        modifier = modifier
            .fillMaxWidth()
            .height(height),
        factory = { context ->
            WebView(context).apply {
                settings.apply {
                    javaScriptEnabled = true
                    loadWithOverviewMode = true
                    useWideViewPort = true
                    cacheMode = WebSettings.LOAD_DEFAULT
                    domStorageEnabled = true
                }
                setBackgroundColor(AndroidColor.TRANSPARENT)
                setLayerType(android.view.View.LAYER_TYPE_HARDWARE, null)
                isVerticalScrollBarEnabled = false
                isHorizontalScrollBarEnabled = false
                overScrollMode = WebView.OVER_SCROLL_NEVER
                addJavascriptInterface(bridge, "AndroidBridge")
                webViewClient = WebViewClient()
                tag = html
                loadDataWithBaseURL("file:///android_asset/", html, "text/html", "UTF-8", null)
            }
        },
        update = { webView ->
            if (webView.tag != html) {
                webView.tag = html
                webView.loadDataWithBaseURL("file:///android_asset/", html, "text/html", "UTF-8", null)
            }
        }
    )
}

private class LatexParagraphBridge(
    private val onHeightChanged: (Int) -> Unit
) {
    private val mainHandler = Handler(Looper.getMainLooper())

    @JavascriptInterface
    fun reportHeight(heightPx: Int) {
        mainHandler.post {
            onHeightChanged(heightPx)
        }
    }
}

private fun buildLatexParagraphHtml(
    spans: List<LatexParagraphSpan>,
    textColor: String,
    fontSizePx: Int,
    fontWeight: Int,
    lineHeight: Float
): String {
    val body = spans.joinToString(separator = "") { span ->
        when (span) {
            is LatexParagraphSpan.Text -> escapeHtml(span.text).replace("\n", "<br>")
            is LatexParagraphSpan.Math -> {
                """<span class="math-inline" data-latex="${escapeHtmlAttribute(span.latex)}"></span>"""
            }
        }
    }

    return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
            <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
            <style>
                * { box-sizing: border-box; }
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    background: transparent;
                    overflow: hidden;
                }
                body {
                    color: $textColor;
                    font-size: ${fontSizePx}px;
                    font-weight: $fontWeight;
                    line-height: $lineHeight;
                    word-break: keep-all;
                    overflow-wrap: break-word;
                }
                #content {
                    width: 100%;
                    padding: 0 0 6px 0;
                    text-align: left;
                    white-space: normal;
                }
                .math-inline {
                    display: inline;
                    white-space: nowrap;
                }
                .katex {
                    font-size: 1em;
                    line-height: 1;
                }
                .katex-display {
                    margin: 0;
                }
            </style>
        </head>
        <body>
            <div id="content">$body</div>
            <script>
                function reportHeight() {
                    var height = Math.ceil(document.documentElement.scrollHeight || document.body.scrollHeight || 1);
                    if (window.AndroidBridge && window.AndroidBridge.reportHeight) {
                        window.AndroidBridge.reportHeight(height + 2);
                    }
                }

                function renderMath() {
                    var elements = document.querySelectorAll('.math-inline');
                    elements.forEach(function(element) {
                        var latex = element.getAttribute('data-latex') || '';
                        try {
                            if (window.katex) {
                                katex.render(latex, element, {
                                    throwOnError: false,
                                    displayMode: false,
                                    output: 'html'
                                });
                            } else {
                                element.textContent = latex;
                            }
                        } catch (e) {
                            element.textContent = latex;
                        }
                    });
                    reportHeight();
                    setTimeout(reportHeight, 50);
                    setTimeout(reportHeight, 150);
                }

                window.addEventListener('load', renderMath);
                renderMath();
                if (document.fonts && document.fonts.ready) {
                    document.fonts.ready.then(reportHeight);
                }
            </script>
        </body>
        </html>
    """.trimIndent()
}

private fun escapeHtml(value: String): String {
    return value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#39;")
}

private fun escapeHtmlAttribute(value: String): String {
    return escapeHtml(value)
}
