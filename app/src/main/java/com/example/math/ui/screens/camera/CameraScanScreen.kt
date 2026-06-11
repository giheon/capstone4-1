package com.example.math.ui.screens.camera

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.util.Base64
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.math.ui.theme.*
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import android.os.Handler
import android.os.Looper
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraScanScreen(
    onBack: () -> Unit,
    onCapture: (String) -> Unit  // base64 이미지 전달
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // 카메라 권한 상태
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    // 권한이 있으면 카메라 화면, 없으면 권한 요청 화면
    when {
        cameraPermissionState.status.isGranted -> {
            CameraContent(
                context = context,
                lifecycleOwner = lifecycleOwner,
                onBack = onBack,
                onCapture = onCapture
            )
        }
        cameraPermissionState.status.shouldShowRationale -> {
            PermissionRationaleScreen(
                onBack = onBack,
                onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
            )
        }
        else -> {
            LaunchedEffect(Unit) {
                cameraPermissionState.launchPermissionRequest()
            }
            PermissionRequestScreen(
                onBack = onBack,
                onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
            )
        }
    }
}

@Composable
private fun CameraContent(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    onBack: () -> Unit,
    onCapture: (String) -> Unit
) {
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var isCapturing by remember { mutableStateOf(false) }
    var isUploading by remember { mutableStateOf(false) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri == null || isUploading) return@rememberLauncherForActivityResult
        isUploading = true
        val base64Image = uriToBase64(context, uri)
        isUploading = false
        if (base64Image != null) {
            onCapture(base64Image)
        } else {
            Log.e("CameraUpload", "Failed to convert selected image")
        }
    }

    // 스캔 라인 애니메이션
    val infiniteTransition = rememberInfiniteTransition(label = "scan")
    val scanLineOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scanLine"
    )

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .padding(top = 80.dp, bottom = 140.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // 안내 텍스트
            Text(
                text = "풀고 싶은 수능 수학 문제를\n영역에 맞춰주세요.",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Blue900,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // 카메라 프리뷰
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .aspectRatio(2f / 3f)
                    .clip(RoundedCornerShape(16.dp))
                    .border(3.dp, Blue500, RoundedCornerShape(16.dp))
            ) {
                // CameraX Preview
                AndroidView(
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }
                    },
                    modifier = Modifier.fillMaxSize(),
                    update = { previewView ->
                        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                        cameraProviderFuture.addListener({
                            val cameraProvider = cameraProviderFuture.get()

                            // Preview
                            val preview = Preview.Builder()
                                .build()
                                .also {
                                    it.surfaceProvider = previewView.surfaceProvider
                                }

                            // ImageCapture
                            imageCapture = ImageCapture.Builder()
                                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                                .build()

                            // 후면 카메라 선택
                            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                            try {
                                cameraProvider.unbindAll()
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    cameraSelector,
                                    preview,
                                    imageCapture
                                )
                            } catch (e: Exception) {
                                Log.e("CameraX", "Use case binding failed", e)
                            }
                        }, ContextCompat.getMainExecutor(context))
                    }
                )

                // 스캔 라인
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(2.dp)
                        .offset(y = (scanLineOffset * 400).dp)
                        .background(
                            brush = Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    Blue500.copy(alpha = 0.8f),
                                    Color.Transparent
                                )
                            )
                        )
                )
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Button(
                onClick = {
                    if (!isUploading) {
                        imagePickerLauncher.launch("image/*")
                    }
                },
                enabled = !isUploading,
                modifier = Modifier
                    .width(176.dp)
                    .height(52.dp),
                shape = RoundedCornerShape(26.dp),
                border = ButtonDefaults.outlinedButtonBorder(enabled = true),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Blue600,
                    disabledContainerColor = Color.White.copy(alpha = 0.75f),
                    disabledContentColor = Gray500
                ),
                elevation = ButtonDefaults.buttonElevation(
                    defaultElevation = 8.dp,
                    pressedElevation = 2.dp,
                    disabledElevation = 0.dp
                ),
                contentPadding = PaddingValues(horizontal = 18.dp)
            ) {
                if (isUploading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Blue600,
                        strokeWidth = 2.dp
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.FileUpload,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                        tint = Blue600
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "이미지 업로드",
                        fontWeight = FontWeight.SemiBold,
                        color = Blue600
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = {
                    if (!isCapturing) {
                        isCapturing = true
                        captureImage(
                            imageCapture = imageCapture,
                            executor = cameraExecutor,
                            onImageCaptured = { base64Image ->
                                isCapturing = false
                                onCapture(base64Image)
                            },
                            onError = { error ->
                                isCapturing = false
                                Log.e("CameraX", "Image capture failed: $error")
                            }
                        )
                    }
                },
                enabled = !isCapturing,
                modifier = Modifier.size(84.dp),
                shape = CircleShape,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isCapturing) Gray300 else Blue500
                ),
                contentPadding = PaddingValues(0.dp)
            ) {
                if (isCapturing) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(32.dp),
                        color = Color.White,
                        strokeWidth = 3.dp
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .size(66.dp)
                            .border(4.dp, Color.White, CircleShape)
                    )
                }
            }
        }

        // 뒤로가기 버튼
        FilledIconButton(
            onClick = onBack,
            modifier = Modifier
                .statusBarsPadding()
                .padding(16.dp)
                .size(48.dp),
            shape = CircleShape,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = Color.White,
                contentColor = Blue600
            )
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "뒤로가기"
            )
        }
    }
}

private fun captureImage(
    imageCapture: ImageCapture?,
    executor: ExecutorService,
    onImageCaptured: (String) -> Unit,
    onError: (String) -> Unit
) {
    val mainHandler = Handler(Looper.getMainLooper())

    imageCapture?.takePicture(
        executor,
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(imageProxy: ImageProxy) {
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                val bitmap = imageProxyToBitmap(imageProxy)
                imageProxy.close()

                if (bitmap != null) {
                    // 이미지 회전 보정
                    val rotatedBitmap = rotateBitmap(bitmap, rotationDegrees)

                    // 이미지 리사이즈 (API 전송을 위해)
                    val resizedBitmap = resizeBitmap(rotatedBitmap, 1024)

                    // Base64 인코딩
                    val base64 = bitmapToBase64(resizedBitmap)

                    // 메인 스레드에서 콜백 호출
                    mainHandler.post {
                        onImageCaptured(base64)
                    }
                } else {
                    mainHandler.post {
                        onError("Failed to convert image")
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                mainHandler.post {
                    onError(exception.message ?: "Unknown error")
                }
            }
        }
    ) ?: onError("ImageCapture not initialized")
}

private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
    val buffer = imageProxy.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return bitmap

    val matrix = Matrix().apply {
        postRotate(rotationDegrees.toFloat())
    }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

private fun resizeBitmap(bitmap: Bitmap, maxSize: Int): Bitmap {
    val width = bitmap.width
    val height = bitmap.height

    val ratio = width.toFloat() / height.toFloat()

    val newWidth: Int
    val newHeight: Int

    if (width > height) {
        newWidth = maxSize
        newHeight = (maxSize / ratio).toInt()
    } else {
        newHeight = maxSize
        newWidth = (maxSize * ratio).toInt()
    }

    return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
}

private fun bitmapToBase64(bitmap: Bitmap): String {
    val outputStream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, outputStream)
    val byteArray = outputStream.toByteArray()
    return Base64.encodeToString(byteArray, Base64.NO_WRAP)
}

private fun uriToBase64(context: Context, uri: Uri): String? {
    return try {
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            val byteArray = inputStream.readBytes()
            Base64.encodeToString(byteArray, Base64.NO_WRAP)
        }
    } catch (e: Exception) {
        Log.e("CameraUpload", "Failed to read selected image", e)
        null
    }
}

@Composable
private fun PermissionRationaleScreen(
    onBack: () -> Unit,
    onRequestPermission: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "카메라 권한이 필요합니다",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Blue900
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "수학 문제를 촬영하기 위해\n카메라 접근 권한이 필요합니다.",
                fontSize = 14.sp,
                color = Gray500,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = onRequestPermission,
                colors = ButtonDefaults.buttonColors(containerColor = Blue500),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text("권한 허용하기", modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
            }

            Spacer(modifier = Modifier.height(16.dp))

            TextButton(onClick = onBack) {
                Text("뒤로가기", color = Gray500)
            }
        }
    }
}

@Composable
private fun PermissionRequestScreen(
    onBack: () -> Unit,
    onRequestPermission: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Blue50)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            CircularProgressIndicator(color = Blue500)

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "카메라 권한을 확인하고 있습니다...",
                fontSize = 14.sp,
                color = Gray500
            )
        }

        // 뒤로가기 버튼
        FilledIconButton(
            onClick = onBack,
            modifier = Modifier
                .statusBarsPadding()
                .padding(16.dp)
                .size(48.dp),
            shape = CircleShape,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = Color.White,
                contentColor = Blue600
            )
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "뒤로가기"
            )
        }
    }
}
