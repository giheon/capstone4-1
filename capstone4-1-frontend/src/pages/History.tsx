import { Clock } from "lucide-react";

export function History() {
  return (
    <div className="min-h-full flex flex-col items-center justify-center text-slate-400 bg-slate-50">
      <Clock size={56} className="mb-4 opacity-20" />
      <p className="font-medium text-slate-500 text-sm">학습 기록이 없습니다.</p>
    </div>
  );
}