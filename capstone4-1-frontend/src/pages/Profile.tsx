import { User } from "lucide-react";

export function Profile() {
  return (
    <div className="min-h-full flex flex-col items-center justify-center text-slate-400 bg-slate-50">
      <User size={56} className="mb-4 opacity-20" />
      <p className="font-medium text-slate-500 text-sm">내 정보 설정 페이지</p>
    </div>
  );
}