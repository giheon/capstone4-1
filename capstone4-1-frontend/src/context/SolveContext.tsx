import { createContext, useContext, useState, type ReactNode } from "react";
import type { SolveResponse } from "../api/types";

interface SolveContextType {
  // 업로드된 이미지 관련
  imageFile: File | null;
  imagePreview: string | null;
  setImage: (file: File | null) => void;
  clearImage: () => void;

  // 문제 텍스트 (선택적)
  questionText: string;
  setQuestionText: (text: string) => void;

  // API 응답 결과
  solveResult: SolveResponse | null;
  setSolveResult: (result: SolveResponse | null) => void;

  // 로딩 및 에러 상태
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;

  // 전체 상태 초기화
  resetAll: () => void;
}

const SolveContext = createContext<SolveContextType | undefined>(undefined);

export function SolveProvider({ children }: { children: ReactNode }) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [questionText, setQuestionText] = useState("");
  const [solveResult, setSolveResult] = useState<SolveResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setImage = (file: File | null) => {
    setImageFile(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  const resetAll = () => {
    setImageFile(null);
    setImagePreview(null);
    setQuestionText("");
    setSolveResult(null);
    setIsLoading(false);
    setError(null);
  };

  return (
    <SolveContext.Provider
      value={{
        imageFile,
        imagePreview,
        setImage,
        clearImage,
        questionText,
        setQuestionText,
        solveResult,
        setSolveResult,
        isLoading,
        setIsLoading,
        error,
        setError,
        resetAll,
      }}
    >
      {children}
    </SolveContext.Provider>
  );
}

export function useSolve() {
  const context = useContext(SolveContext);
  if (context === undefined) {
    throw new Error("useSolve must be used within a SolveProvider");
  }
  return context;
}