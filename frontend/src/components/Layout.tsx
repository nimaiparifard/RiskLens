import { Activity, Boxes, FileText, LogOut, Moon, ShieldAlert, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Navigate, NavLink, Outlet, useLocation } from "react-router-dom";

import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/applications", label: "Applications", icon: FileText },
  { to: "/fraud", label: "Fraud monitor", icon: ShieldAlert },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/monitoring", label: "Monitoring", icon: Activity },
];

function useTheme() {
  const [theme, setTheme] = useState<string | null>(() => {
    try {
      return localStorage.getItem("risklens.theme");
    } catch {
      return null;
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem("risklens.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  const dark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return { dark, toggle: () => setTheme(dark ? "light" : "dark") };
}

export function ProtectedLayout() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const { dark, toggle } = useTheme();
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="flex shrink-0 flex-row items-center gap-1 border-b border-[var(--border)] bg-surface p-3 md:w-56 md:flex-col md:items-stretch md:border-b-0 md:border-r">
        <div className="mb-0 mr-4 flex items-center gap-2 px-2 md:mb-6 md:mr-0">
          <span className="inline-block h-6 w-6 rounded-full border-[6px] border-brand" aria-hidden />
          <span className="text-base font-semibold text-ink">RiskLens</span>
        </div>
        <nav className="flex flex-1 flex-row gap-1 overflow-x-auto md:flex-col">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 whitespace-nowrap rounded-md px-3 py-2 text-sm",
                  isActive ? "bg-[var(--gridline)] font-medium text-ink" : "text-ink-2 hover:bg-[var(--gridline)]",
                )
              }
            >
              <Icon className="h-4 w-4" aria-hidden /> <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-1 md:mt-auto md:flex-col md:items-stretch">
          <div className="hidden px-3 py-2 text-xs text-ink-2 md:block">
            {user.username} · <span className="font-medium">{user.role}</span>
          </div>
          <button onClick={toggle} className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-ink-2 hover:bg-[var(--gridline)]" aria-label="Toggle theme">
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            <span className="hidden md:inline">{dark ? "Light" : "Dark"} mode</span>
          </button>
          <button onClick={logout} className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-ink-2 hover:bg-[var(--gridline)]">
            <LogOut className="h-4 w-4" aria-hidden /> <span className="hidden md:inline">Sign out</span>
          </button>
        </div>
      </aside>
      <main className="min-w-0 flex-1 p-4 md:p-8">
        <Outlet />
      </main>
    </div>
  );
}
