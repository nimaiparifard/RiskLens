import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { Button, Card, ErrorBox, Input, Label } from "@/components/ui";
import { useAuth } from "@/lib/auth";

export function LoginPage() {
  const { user, login, loading } = useAuth();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);

  if (user) {
    const from = (location.state as { from?: string } | null)?.from ?? "/applications";
    return <Navigate to={from} replace />;
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await login(username, password);
    } catch (err) {
      setError(err);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-sm p-6">
        <div className="mb-6 flex items-center gap-2">
          <span className="inline-block h-7 w-7 rounded-full border-[7px] border-brand" aria-hidden />
          <div>
            <h1 className="text-lg font-semibold text-ink">RiskLens</h1>
            <p className="text-xs text-ink-2">Credit risk & fraud scoring</p>
          </div>
        </div>
        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <Label htmlFor="username">Username</Label>
            <Input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
          </div>
          <div>
            <Label htmlFor="password">Password</Label>
            <Input id="password" type="password" autoComplete="current-password" value={password}
              onChange={(e) => setPassword(e.target.value)} required />
          </div>
          {error != null && <ErrorBox error={error} />}
          <Button type="submit" className="w-full" loading={loading}>
            Sign in
          </Button>
          <p className="text-center text-xs text-ink-muted">Demo users: officer · admin · viewer</p>
        </form>
      </Card>
    </div>
  );
}
