package com.example.math.data

/**
 * 촬영된 이미지를 임시로 저장하는 싱글톤 객체
 * 카메라에서 촬영한 이미지를 다음 화면으로 전달할 때 사용
 */
object CapturedImageHolder {
    private var _imageBase64: String? = null

    val imageBase64: String?
        get() = _imageBase64

    fun setImage(base64: String) {
        _imageBase64 = base64
    }

    fun clear() {
        _imageBase64 = null
    }

    fun hasImage(): Boolean = _imageBase64 != null
}
