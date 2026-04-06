// 백엔드 API 응답 타입 정의

export interface Concept {
  concept_id: string;
  label_ko: string;
  subject: string;
  chapter_id: string;
  type: "concept";
}

export interface FlowStep {
  order: number;
  concept_id: string;
  label_ko: string;
}

export interface SimilarProblem {
  problem_id: string;
  source_org: string;
  year: number;
  month: number | null;
  subject: string;
  question_no: number | null;
  stem: string;  // 문제 본문
  answer: string;
}

export interface SimilarProblemsResponse {
  status: "ok" | "pending_corpus" | "error";
  items: SimilarProblem[];
}

export interface SolveResponse {
  request_id: string;
  difficulty: "상" | "중" | "하";
  selected_model: string;
  extracted_problem: string;  // OCR로 추출된 문제 (LaTeX)
  solution: string;
  answer: string;
  concepts: Concept[];
  flow: FlowStep[];
  similar_problems: SimilarProblemsResponse;
}

export interface ErrorResponse {
  request_id: string;
  error_code: string;
  message: string;
}

export interface HealthResponse {
  status: "ok";
}