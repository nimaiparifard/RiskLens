import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { setUnauthorizedHandler, tokens } from "@/api/client";
import { api } from "@/api/endpoints";
import type { User } from "@/api/schemas";

type AuthState = {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  can: (action: "decide" | "admin") => boolean;
};

const AuthContext = createContext<AuthState | null>(null);
const USER_KEY = "risklens.user";

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [user, setUser] = useState<User | null>(() => {
    const raw = localStorage.getItem(USER_KEY);
    return raw && tokens.access ? (JSON.parse(raw) as User) : null;
  });
  const [loading, setLoading] = useState(false);

  const logout = useCallback(() => {
    tokens.clear();
    localStorage.removeItem(USER_KEY);
    setUser(null);
    qc.clear();
  }, [qc]);

  useEffect(() => setUnauthorizedHandler(logout), [logout]);

  const login = useCallback(async (username: string, password: string) => {
    setLoading(true);
    try {
      const res = await api.login(username, password);
      tokens.set(res.access, res.refresh);
      localStorage.setItem(USER_KEY, JSON.stringify(res.user));
      setUser(res.user);
    } finally {
      setLoading(false);
    }
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      user,
      loading,
      login,
      logout,
      can: (action) =>
        action === "admin" ? user?.role === "admin" : user?.role === "officer" || user?.role === "admin",
    }),
    [user, loading, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
