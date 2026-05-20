package com.example.math.ui.components

import android.annotation.SuppressLint
import android.graphics.Color
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.math.ui.theme.Gray100
import java.net.URLEncoder

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun LatexFormula(
    latex: String,
    modifier: Modifier = Modifier,
    onRendered: () -> Unit = {}
) {
    var hasRendered by remember { mutableStateOf(false) }
    val encodedLatex = remember(latex) {
        URLEncoder.encode(latex, "UTF-8")
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Gray100),
        contentAlignment = Alignment.Center
    ) {
        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .wrapContentHeight()
                .defaultMinSize(minHeight = 36.dp),
            factory = { context ->
                WebView(context).apply {
                    settings.apply {
                        javaScriptEnabled = true
                        loadWithOverviewMode = true
                        useWideViewPort = true
                        cacheMode = WebSettings.LOAD_DEFAULT
                        domStorageEnabled = true
                    }
                    setBackgroundColor(Color.TRANSPARENT)
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

                    loadUrl("file:///android_asset/katex.html?latex=$encodedLatex")
                }
            },
            update = { webView ->
                webView.loadUrl("file:///android_asset/katex.html?latex=$encodedLatex")
            }
        )
    }
}