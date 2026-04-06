import { useState, useEffect, useRef } from "react";
import { motion } from "motion/react";
import { useNavigate } from "react-router";
import { ScanFace, Cpu, CheckCircle, AlertCircle } from "lucide-react";
import { useSolve } from "../context/SolveContext";
import { solveProblem } from "../api/mathSolver";

const FALLBACK_IMAGE = "https://images.unsplash.com/photo-1560785472-2f186f554644?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtYXRoJTIwcHJvYmxlbSUyMGhhbmR3cml0dGVuJTIwb24lMjBwYXBlcnxlbnwxfHx8fDE3NzQ5MzMwMzd8MA&ixlib=rb-4.1.0&q=80&w=1080";

export function Processing() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const apiCalled = useRef(false);

  const { imageFile, imagePreview, questionText, setSolveResult, setError: setGlobalError } = useSolve();

  const steps = [
    { text: "수식 추출 중 (OCR)...", icon: ScanFace },
    { text: "난이도 분석 중 (GPT-4o)...", icon: Cpu },
    { text: "풀이 생성 중...", icon: CheckCircle }
  ];

  useEffect(() => {
    // 이미지가 없으면 홈으로 리다이렉트
    if (!imageFile && !questionText) {
      navigate("/");
      return;
    }

    // API 중복 호출 방지
    if (apiCalled.current) return;
    apiCalled.current = true;

    const callApi = async () => {
      try {
        // 단계별 진행 표시 (UI 피드백용)
        setStep(0);
        console.log("🚀 API 호출 시작...");

        // API 호출 시작
        const result = await solveProblem(questionText || undefined, imageFile || undefined);
        console.log("✅ API 응답 수신:", result);

        // 단계 2로 진행
        setStep(1);
        await new Promise(resolve => setTimeout(resolve, 500));

        // 단계 3으로 진행
        setStep(2);
        await new Promise(resolve => setTimeout(resolve, 500));

        // 결과 저장
        console.log("💾 결과 저장 중...");
        setSolveResult(result);
        console.log("✅ 결과 저장 완료, 결과 페이지로 이동");

        // 결과 페이지로 이동
        navigate("/result");
      } catch (err) {
        console.error("❌ API 에러 발생:", err);
        const message = err instanceof Error ? err.message : "문제 풀이 중 오류가 발생했습니다.";
        setError(message);
        setGlobalError(message);
      }
    };

    // 애니메이션을 위한 약간의 딜레이 후 API 호출
    const timer = setTimeout(() => {
      callApi();
    }, 500);

    return () => clearTimeout(timer);
  }, [imageFile, questionText, navigate, setSolveResult, setGlobalError]);

  // 진행 상태에 따른 시각적 업데이트 (API 호출 중에도)
  useEffect(() => {
    if (error) return;

    const interval = setInterval(() => {
      setStep(prev => (prev < 2 ? prev : prev));
    }, 1500);

    return () => clearInterval(interval);
  }, [error]);

  const displayImage = imagePreview || FALLBACK_IMAGE;

  if (error) {
    return (
      <div className="h-screen w-full max-w-md mx-auto bg-slate-900 relative overflow-hidden flex flex-col items-center justify-center p-8">
        <div className="bg-red-500/20 border border-red-500/50 rounded-3xl p-8 text-center">
          <AlertCircle size={48} className="text-red-400 mx-auto mb-4" />
          <h2 className="text-white font-bold text-lg mb-2">오류 발생</h2>
          <p className="text-red-200 text-sm mb-6">{error}</p>
          <button
            onClick={() => navigate("/")}
            className="px-6 py-3 bg-white text-slate-900 font-bold rounded-xl hover:bg-slate-100 transition-colors"
          >
            다시 시도하기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full max-w-md mx-auto bg-slate-900 relative overflow-hidden flex flex-col items-center justify-center">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0 opacity-20 blur-sm">
        <img src={displayImage} alt="Math Problem" className="w-full h-full object-cover" />
      </div>
      <div className="absolute inset-0 bg-slate-900/60 z-10" />

      {/* Main Content */}
      <div className="z-20 flex flex-col items-center w-full px-8">
        {/* Scanning Animation Area */}
        <div className="relative w-64 h-64 rounded-2xl border border-blue-500/30 overflow-hidden mb-12 shadow-[0_0_40px_rgba(59,130,246,0.25)] bg-slate-800/50 backdrop-blur-md">
          <img src={displayImage} alt="Scan Target" className="w-full h-full object-cover opacity-60" />

          <motion.div
            className="absolute top-0 left-0 w-full h-[2px] bg-blue-400 shadow-[0_0_20px_rgba(96,165,250,1)]"
            animate={{ top: ["0%", "100%", "0%"] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 to-transparent pointer-events-none" />

          {/* Circular Indicator inside */}
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/20">
            <svg className="w-24 h-24 transform -rotate-90">
              <circle cx="48" cy="48" r="40" stroke="rgba(255,255,255,0.05)" strokeWidth="6" fill="none" />
              <motion.circle
                cx="48" cy="48" r="40" stroke="#3b82f6" strokeWidth="6" fill="none" strokeLinecap="round"
                initial={{ strokeDasharray: 251.2, strokeDashoffset: 251.2 }}
                animate={{ strokeDashoffset: 251.2 - (251.2 * ((step + 1) / 3)) }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
              />
            </svg>
            <div className="absolute font-bold text-white text-xl flex items-center gap-1">
              {Math.round(((step + 1) / 3) * 100)}<span className="text-sm text-blue-300">%</span>
            </div>
          </div>
        </div>

        {/* Real-time Status Updates */}
        <div className="w-full bg-slate-800/80 backdrop-blur-lg rounded-3xl p-6 border border-slate-700/50 shadow-2xl">
          <div className="flex flex-col gap-5">
            {steps.map((s, idx) => {
              const isActive = idx === step;
              const isPast = idx < step;
              const Icon = s.icon;
              return (
                <div key={idx} className={`flex items-center gap-4 transition-all duration-500 ${isActive ? "opacity-100 transform translate-x-2" : isPast ? "opacity-40" : "opacity-20"}`}>
                  <div className={`p-2 rounded-xl transition-all duration-300 ${isActive ? "bg-blue-500 text-white shadow-[0_0_20px_rgba(59,130,246,0.6)]" : "bg-slate-700 text-slate-400"}`}>
                    <Icon size={20} strokeWidth={isActive ? 2.5 : 2} />
                  </div>
                  <span className={`font-semibold text-sm ${isActive ? "text-blue-50" : "text-slate-400"}`}>
                    {s.text}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}