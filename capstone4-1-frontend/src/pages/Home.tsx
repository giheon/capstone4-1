import { useRef, useState } from "react";
import { Camera, Upload, X, Loader2 } from "lucide-react";
import { useNavigate } from "react-router";
import { useSolve } from "../context/SolveContext";

export function Home() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { imageFile, imagePreview, setImage, clearImage } = useSolve();
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith("image/")) {
        setImage(file);
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setImage(file);
      }
    }
  };

  const handleAnalyze = () => {
    if (imageFile) {
      navigate("/processing");
    }
  };

  return (
    <div className="flex flex-col h-full bg-white px-6 pt-12 pb-8">
      <h1 className="text-2xl font-bold text-slate-900 mb-2">문제 풀이 분석</h1>
      <p className="text-slate-500 mb-8 text-sm">수학 문제 사진을 업로드하면 AI가 인식하여 풀이 과정을 제공해 드립니다.</p>

      {imagePreview ? (
        // 이미지가 업로드된 경우
        <div className="flex-1 flex flex-col">
          <div className="relative flex-1 rounded-3xl overflow-hidden border border-slate-200 bg-slate-50">
            <img
              src={imagePreview}
              alt="업로드된 문제"
              className="w-full h-full object-contain"
            />
            <button
              onClick={clearImage}
              className="absolute top-3 right-3 p-2 bg-red-500 text-white rounded-full shadow-lg hover:bg-red-600 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          <button
            onClick={handleAnalyze}
            className="mt-4 w-full py-4 bg-blue-600 text-white font-bold rounded-2xl hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <Loader2 size={20} className="animate-none" />
            AI 분석 시작하기
          </button>
        </div>
      ) : (
        // 이미지 업로드 영역
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`flex-1 border-2 border-dashed rounded-3xl flex flex-col items-center justify-center cursor-pointer transition-colors ${
            dragActive
              ? "border-blue-400 bg-blue-100/50"
              : "border-blue-200 bg-blue-50/50 hover:bg-blue-50"
          }`}
        >
          <div className="bg-white p-5 rounded-full shadow-sm mb-4 border border-blue-100">
            <Camera size={36} className="text-blue-600" />
          </div>
          <h3 className="font-bold text-blue-900 text-lg">사진 찍기 또는 업로드</h3>
          <p className="text-sm text-blue-600/70 mt-1">탭해서 수학 문제를 촬영하세요</p>
          <p className="text-xs text-slate-400 mt-4">JPG, PNG, WebP (최대 10MB)</p>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
}