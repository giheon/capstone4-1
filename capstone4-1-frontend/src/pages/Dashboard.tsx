import { useState } from "react";
import { BookOpen, Target, PlayCircle, BarChart2, ChevronUp } from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";

const BlockMath = ({ math }: { math: string }) => {
  return (
    <div 
      className="py-1"
      dangerouslySetInnerHTML={{ 
        __html: katex.renderToString(math, { displayMode: true, throwOnError: false }) 
      }} 
    />
  );
};

export function Dashboard() {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const matchedProblems = [
    { 
      id: 1, topic: "정적분의 기본 정리", formula: "\\int_0^3 (x^2 - 2x) dx", difficulty: "Easy", diffColor: "bg-emerald-100 text-emerald-700", score: 92,
      solution: [
        "\\int (x^2 - 2x) dx = \\frac{1}{3}x^3 - x^2",
        "\\left[\\frac{1}{3}x^3 - x^2\\right]_0^3 = \\left(\\frac{27}{3} - 9\\right) - 0",
        "= 9 - 9 = 0"
      ]
    },
    { 
      id: 2, topic: "정적분과 넓이", formula: "\\int_{-1}^2 (x^2 - 4x + 3) dx", difficulty: "Normal", diffColor: "bg-blue-100 text-blue-700", score: 85,
      solution: [
        "\\int (x^2 - 4x + 3) dx = \\frac{1}{3}x^3 - 2x^2 + 3x",
        "\\left[\\frac{1}{3}x^3 - 2x^2 + 3x\\right]_{-1}^2",
        "= \\left(\\frac{8}{3} - 8 + 6\\right) - \\left(-\\frac{1}{3} - 2 - 3\\right)",
        "= \\frac{2}{3} - \\left(-\\frac{16}{3}\\right) = \\frac{18}{3} = 6"
      ]
    },
    { 
      id: 3, topic: "치환적분법", formula: "\\int_0^1 x(x^2+1)^3 dx", difficulty: "Hard", diffColor: "bg-orange-100 text-orange-700", score: 68,
      solution: [
        "x^2+1 = t \\text{ 로 치환하면, } 2x dx = dt",
        "\\int_1^2 t^3 \\frac{1}{2} dt = \\left[\\frac{1}{8}t^4\\right]_1^2",
        "= \\frac{16}{8} - \\frac{1}{8} = \\frac{15}{8}"
      ]
    },
  ];

  return (
    <div className="min-h-full bg-slate-50 flex flex-col pb-6">
      <div className="bg-white px-6 pt-10 pb-6 shadow-sm border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 mb-1.5 tracking-tight">유사 문제 매칭</h1>
        <p className="text-sm text-slate-500 font-medium">방금 학습한 내용과 관련된 맞춤형 추천 문제입니다.</p>
      </div>

      <div className="p-5 flex-1 space-y-8">
        {/* Current Problem Context */}
        <div className="bg-gradient-to-br from-blue-600 to-indigo-800 rounded-3xl p-6 text-white shadow-xl shadow-blue-900/10 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-2xl -translate-y-10 translate-x-10" />
          
          <div className="flex items-center gap-2 text-blue-100 text-[11px] font-bold mb-4 uppercase tracking-wider relative z-10">
            <BookOpen size={14} strokeWidth={2.5} /> Current Study
          </div>
          <div className="mb-1.5 relative z-10 bg-white/10 inline-block px-4 py-1 rounded-xl border border-white/20 backdrop-blur-sm overflow-x-auto max-w-full text-white">
            <BlockMath math="\int_{0}^{2} (3x^2 - 2x + 1) dx" />
          </div>
          <p className="text-[13px] text-blue-100/90 font-medium relative z-10 mt-3 pl-1">정해진 구간에서의 다항함수 정적분</p>
        </div>

        {/* DB Match List */}
        <div>
          <div className="flex items-center justify-between mb-5 px-1">
            <h2 className="font-bold text-slate-900 flex items-center gap-2 text-[15px]">
              <div className="p-1.5 bg-indigo-100 rounded-lg text-indigo-600">
                <Target size={16} strokeWidth={2.5} />
              </div>
              DB 맞춤 매칭 목록
            </h2>
            <span className="text-[11px] font-bold text-slate-500 bg-slate-200/80 px-2.5 py-1 rounded-full">{matchedProblems.length}건</span>
          </div>

          <div className="space-y-4">
            {matchedProblems.map((prob) => {
              const isExpanded = expandedId === prob.id;
              
              return (
                <div 
                  key={prob.id} 
                  className={`bg-white border ${isExpanded ? 'border-blue-300 shadow-[0_8px_24px_rgba(59,130,246,0.12)]' : 'border-slate-200/60 shadow-[0_4px_12px_rgba(0,0,0,0.02)]'} rounded-3xl p-5 flex flex-col transition-all cursor-pointer group`}
                >
                  <div className="flex justify-between items-start mb-3" onClick={() => setExpandedId(isExpanded ? null : prob.id)}>
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-bold px-2.5 py-1 rounded-full ${prob.diffColor}`}>
                        {prob.difficulty}
                      </span>
                      <span className="text-xs font-bold text-slate-600">{prob.topic}</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-[11px] font-bold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-100/50">
                      <BarChart2 size={12} strokeWidth={2.5} />
                      {prob.score}% 매칭
                    </div>
                  </div>
                  
                  <div 
                    className="text-[14px] text-slate-800 mb-5 mt-1 bg-slate-50/80 p-1.5 rounded-xl border border-slate-100 overflow-x-auto overflow-y-hidden"
                    onClick={() => setExpandedId(isExpanded ? null : prob.id)}
                  >
                    <BlockMath math={prob.formula} />
                  </div>

                  {/* Expanded Solution Section */}
                  {isExpanded && (
                    <div className="mb-5 p-4 bg-blue-50/60 rounded-2xl border border-blue-100/60">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-[11px] font-bold text-blue-800 uppercase tracking-wider flex items-center gap-1">
                          <BookOpen size={12} /> DB 저장 해설
                        </h4>
                      </div>
                      <div className="space-y-4">
                        {prob.solution.map((step, idx) => (
                          <div key={idx} className="flex gap-3 items-center">
                            <div className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-[10px] font-bold shrink-0">{idx + 1}</div>
                            <div className="text-[13px] text-slate-700 w-full overflow-x-auto overflow-y-hidden bg-white/50 px-2 py-1 rounded-lg">
                              <BlockMath math={step} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <button 
                    onClick={() => setExpandedId(isExpanded ? null : prob.id)}
                    className={`flex items-center justify-center gap-2 w-full text-sm font-bold py-3 rounded-xl transition-colors ${
                      isExpanded 
                        ? 'bg-slate-100 text-slate-600 hover:bg-slate-200' 
                        : 'bg-slate-900 text-white hover:bg-blue-600 shadow-md shadow-slate-900/10'
                    }`}
                  >
                    {isExpanded ? (
                      <>닫기 <ChevronUp size={18} strokeWidth={2.5} /></>
                    ) : (
                      <><PlayCircle size={18} strokeWidth={2} /> Solve Now (해설 보기)</>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}