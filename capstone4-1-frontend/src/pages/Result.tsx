import { ArrowLeft, ChevronRight, Activity, Cpu, Sparkles, ScanFace, BookOpen, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router";
import katex from "katex";
import "katex/dist/katex.min.css";
import { useSolve } from "../context/SolveContext";

const BlockMath = ({ math }: { math: string }) => {
  return (
    <div
      className="py-2"
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(math, { displayMode: true, throwOnError: false })
      }}
    />
  );
};

// 텍스트에서 LaTeX 수식을 추출하여 렌더링
const RenderSolution = ({ text }: { text: string }) => {
  // $...$ 또는 $$...$$ 패턴 찾기
  const parts = text.split(/(\$\$?[^$]+\$\$?)/g);

  return (
    <div className="text-[13px] text-slate-700 leading-relaxed whitespace-pre-wrap">
      {parts.map((part, idx) => {
        if (part.startsWith("$$") && part.endsWith("$$")) {
          const math = part.slice(2, -2);
          return <BlockMath key={idx} math={math} />;
        } else if (part.startsWith("$") && part.endsWith("$")) {
          const math = part.slice(1, -1);
          return (
            <span
              key={idx}
              dangerouslySetInnerHTML={{
                __html: katex.renderToString(math, { displayMode: false, throwOnError: false })
              }}
            />
          );
        }
        return <span key={idx}>{part}</span>;
      })}
    </div>
  );
};

const FALLBACK_IMAGE = "https://images.unsplash.com/photo-1560785472-2f186f554644?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtYXRoJTIwcHJvYmxlbSUyMGhhbmR3cml0dGVuJTIwb24lMjBwYXBlcnxlbnwxfHx8fDE3NzQ5MzMwMzd8MA&ixlib=rb-4.1.0&q=80&w=1080";

export function Result() {
  const navigate = useNavigate();
  const { solveResult, imagePreview, resetAll } = useSolve();

  const handleGoHome = () => {
    resetAll();
    navigate("/");
  };

  // 결과가 없으면 홈으로
  if (!solveResult) {
    return (
      <div className="h-screen w-full max-w-md mx-auto bg-slate-50 flex flex-col items-center justify-center p-8">
        <AlertCircle size={48} className="text-slate-400 mb-4" />
        <p className="text-slate-600 text-center mb-6">분석 결과가 없습니다.</p>
        <button
          onClick={handleGoHome}
          className="px-6 py-3 bg-blue-600 text-white font-bold rounded-xl hover:bg-blue-700 transition-colors"
        >
          문제 업로드하기
        </button>
      </div>
    );
  }

  const { difficulty, selected_model, extracted_problem, solution, answer, concepts, flow, similar_problems } = solveResult;

  // 난이도별 색상
  const difficultyConfig = {
    "상": { label: "어려움", color: "bg-red-50 text-red-600 border-red-100" },
    "중": { label: "보통", color: "bg-yellow-50 text-yellow-600 border-yellow-100" },
    "하": { label: "쉬움", color: "bg-green-50 text-green-600 border-green-100" },
  };
  const difficultyInfo = difficultyConfig[difficulty] || difficultyConfig["중"];
  const displayImage = imagePreview || FALLBACK_IMAGE;

  return (
    <div className="h-screen w-full max-w-md mx-auto bg-slate-50 overflow-y-auto flex flex-col pb-10 hide-scrollbar relative">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md px-6 py-4 sticky top-0 z-50 border-b border-slate-100 flex items-center justify-between">
        <button onClick={handleGoHome} className="p-2 -ml-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-full transition-colors">
          <ArrowLeft size={22} />
        </button>
        <h1 className="font-bold text-slate-800">분석 리포트</h1>
        <div className="w-8" />
      </div>

      <div className="p-5 space-y-6">
        {/* 1. 문제 이미지 & OCR 추출 결과 */}
        <div className="bg-white rounded-3xl shadow-sm border border-slate-200/60 overflow-hidden">
          <div className="p-5 border-b border-slate-50 flex items-start gap-5">
            <div className="w-20 h-20 rounded-xl overflow-hidden shrink-0 border border-slate-200 bg-slate-100 relative shadow-inner">
              <img src={displayImage} alt="Original" className="w-full h-full object-cover" />
              <div className="absolute inset-0 shadow-[inset_0_0_10px_rgba(0,0,0,0.1)] rounded-xl" />
            </div>
            <div className="w-full min-w-0">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-blue-600 mb-2.5 uppercase tracking-wider">
                <ScanFace size={14} strokeWidth={2.5} /> OCR 추출 결과
              </div>
              {/* OCR 추출된 문제 표시 */}
              {extracted_problem && (
                <div className="text-slate-800 bg-slate-50 px-3 py-2 rounded-lg border border-slate-100 overflow-x-auto mb-2">
                  <RenderSolution text={extracted_problem} />
                </div>
              )}
              <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">
                <span className="text-blue-500 font-semibold">정답:</span> {answer}
              </p>
            </div>
          </div>

          {/* AI Model Info + Difficulty */}
          <div className="bg-blue-50/50 p-4 px-5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Cpu size={16} className="text-blue-500" />
                <span className="text-[11px] font-bold text-slate-600 uppercase tracking-wide">{selected_model}</span>
              </div>
              <span className={`text-[10px] font-bold px-2.5 py-1 rounded-full border ${difficultyInfo.color}`}>
                난이도: {difficultyInfo.label}
              </span>
            </div>
            <div className="flex items-center gap-1.5 bg-white px-3 py-1.5 rounded-full border border-blue-100 shadow-sm">
              <Activity size={14} className="text-green-500" />
              <span className="text-xs font-bold text-slate-800">분석 완료</span>
            </div>
          </div>
        </div>

        {/* 2. 사용된 개념 */}
        {concepts.length > 0 && (
          <div className="bg-white rounded-3xl shadow-sm border border-slate-200/60 p-5">
            <h2 className="flex items-center gap-2 font-bold text-slate-900 mb-4">
              <div className="p-1.5 bg-purple-100 rounded-lg text-purple-600">
                <BookOpen size={16} />
              </div>
              사용된 수학 개념
            </h2>
            <div className="flex flex-wrap gap-2">
              {concepts.map((concept) => (
                <span
                  key={concept.concept_id}
                  className="text-xs font-medium px-3 py-1.5 rounded-full bg-purple-50 text-purple-700 border border-purple-100"
                >
                  {concept.label_ko}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* 3. 풀이 과정 */}
        <div className="bg-white rounded-3xl shadow-sm border border-slate-200/60 p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-blue-50 rounded-bl-[100px] opacity-50 -z-0" />

          <h2 className="flex items-center gap-2 font-bold text-slate-900 mb-5 relative z-10">
            <div className="p-1.5 bg-blue-100 rounded-lg text-blue-600">
              <Sparkles size={18} />
            </div>
            AI 단계별 풀이
          </h2>

          <div className="space-y-6 relative z-10">
            {flow.length > 0 ? (
              // 단계별 흐름이 있는 경우
              flow.map((step, idx) => (
                <div key={step.order} className="flex gap-4">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-black shrink-0 border shadow-sm mt-0.5 ${
                    idx === flow.length - 1
                      ? "bg-blue-600 text-white shadow-blue-600/20"
                      : "bg-blue-50 text-blue-600 border-blue-100"
                  }`}>
                    {step.order}
                  </div>
                  <div className="w-full min-w-0">
                    <p className="text-[13px] text-slate-900 font-bold mb-1.5">{step.label_ko}</p>
                  </div>
                </div>
              ))
            ) : null}

            {/* 전체 풀이 텍스트 */}
            <div className="bg-slate-50/80 p-4 rounded-xl border border-slate-100">
              <RenderSolution text={solution} />
            </div>

            {/* 최종 답 */}
            <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
              <p className="text-sm font-bold text-blue-900">
                정답: <span className="text-lg">{answer}</span>
              </p>
            </div>
          </div>
        </div>

        {/* 4. 유사 문제 */}
        {similar_problems.status === "ok" && similar_problems.items.length > 0 && (
          <div className="pt-2 pb-6">
            <div className="flex items-center justify-between mb-4 px-2">
              <h2 className="font-bold text-slate-900 text-[15px]">추천 유사 문제</h2>
              <button onClick={() => navigate("/dashboard")} className="text-xs text-blue-600 font-bold flex items-center bg-blue-50 px-3 py-1.5 rounded-full hover:bg-blue-100 transition-colors">
                더보기 <ChevronRight size={14} className="ml-0.5" />
              </button>
            </div>

            <div className="flex overflow-x-auto gap-4 pb-6 snap-x hide-scrollbar -mx-5 px-5">
              {similar_problems.items.map((problem, idx) => (
                <div key={idx} className="min-w-[220px] bg-white border border-slate-200/60 rounded-3xl p-5 shadow-[0_4px_12px_rgba(0,0,0,0.03)] snap-center shrink-0 flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-slate-100 text-slate-600">
                        {problem.subject || "수학"}
                      </span>
                      <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-blue-50 text-blue-600">
                        {problem.source_org} {problem.year}
                      </span>
                    </div>
                    <p className="text-[13px] text-slate-800 mb-5 leading-relaxed">
                      {problem.stem}
                    </p>
                  </div>
                  <button className="w-full text-xs font-bold text-center text-blue-600 bg-blue-50 hover:bg-blue-100 py-2.5 rounded-xl transition-colors mt-auto border border-blue-100">
                    풀어보기
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 새 문제 풀기 버튼 */}
        <div className="pt-4">
          <button
            onClick={handleGoHome}
            className="w-full py-4 bg-blue-600 text-white font-bold rounded-2xl hover:bg-blue-700 transition-colors"
          >
            새 문제 분석하기
          </button>
        </div>
      </div>
    </div>
  );
}