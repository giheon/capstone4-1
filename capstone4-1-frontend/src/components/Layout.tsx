import { Outlet, useNavigate, useLocation } from "react-router";
import { Home, Layers, Clock, User } from "lucide-react";

export function Layout() {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { label: "홈", path: "/", icon: Home },
    { label: "학습", path: "/dashboard", icon: Layers },
    { label: "기록", path: "/history", icon: Clock },
    { label: "내 정보", path: "/profile", icon: User },
  ];

  return (
    <div className="flex flex-col h-screen max-w-md mx-auto bg-slate-50 shadow-2xl overflow-hidden relative border-x border-slate-200">
      <main className="flex-1 overflow-y-auto hide-scrollbar">
        <Outlet />
      </main>
      <nav className="bg-white border-t border-slate-200 flex justify-between px-6 py-3 pb-safe z-50">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`flex flex-col items-center gap-1.5 transition-colors ${isActive ? "text-blue-600" : "text-slate-400 hover:text-slate-600"}`}
            >
              <Icon size={24} strokeWidth={isActive ? 2.5 : 2} className={isActive ? "fill-blue-50" : ""} />
              <span className={`text-[10px] ${isActive ? "font-bold" : "font-medium"}`}>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}