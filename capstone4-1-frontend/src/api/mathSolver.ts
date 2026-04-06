import type { SolveResponse, HealthResponse, ErrorResponse } from "./types";

// 백엔드 API 기본 URL (개발 환경)
// Android WebView에서는 localhost가 에뮬레이터 자체를 가리키므로
// 실제 호스트 머신의 백엔드에 접근하려면 10.0.2.2 사용 (에뮬레이터용)
// 또는 adb reverse로 포트 포워딩 설정 필요
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public statusCode: number,
    public errorCode: string,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * 서버 상태 확인
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`);

  if (!response.ok) {
    throw new Error("서버에 연결할 수 없습니다.");
  }

  return response.json();
}

/**
 * 수학 문제 풀이 요청
 * @param questionText - 문제 텍스트 (선택)
 * @param imageFile - 문제 이미지 파일 (선택)
 * @returns 풀이 결과
 */
export async function solveProblem(
  questionText?: string,
  imageFile?: File
): Promise<SolveResponse> {
  if (!questionText && !imageFile) {
    throw new Error("문제 텍스트 또는 이미지 중 하나는 반드시 제공해야 합니다.");
  }

  const formData = new FormData();

  if (questionText) {
    formData.append("question_text", questionText);
  }

  if (imageFile) {
    formData.append("image", imageFile);
  }

  console.log(`📡 API 요청: ${API_BASE_URL}/api/v1/solve`);

  const response = await fetch(`${API_BASE_URL}/api/v1/solve`, {
    method: "POST",
    body: formData,
  });

  console.log(`📥 응답 상태: ${response.status} ${response.statusText}`);

  if (!response.ok) {
    const errorData: ErrorResponse = await response.json().catch(() => ({
      request_id: "",
      error_code: "UNKNOWN_ERROR",
      message: "알 수 없는 오류가 발생했습니다.",
    }));
    console.error("❌ API 에러:", errorData);

    throw new ApiError(
      response.status,
      errorData.error_code,
      errorData.message
    );
  }

  const data = await response.json();
  console.log("📦 API 응답 데이터:", data);
  return data;
}

/**
 * 난이도에 따른 한글 레이블 반환
 */
export function getDifficultyLabel(difficulty: "상" | "중" | "하"): string {
  const labels = {
    "상": "어려움",
    "중": "보통",
    "하": "쉬움",
  };
  return labels[difficulty];
}

/**
 * 난이도에 따른 색상 클래스 반환
 */
export function getDifficultyColor(difficulty: "상" | "중" | "하"): string {
  const colors = {
    "상": "bg-red-50 text-red-600",
    "중": "bg-yellow-50 text-yellow-600",
    "하": "bg-green-50 text-green-600",
  };
  return colors[difficulty];
}

export { ApiError };