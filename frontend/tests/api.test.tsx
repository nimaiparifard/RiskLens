import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { z } from "zod";

import { ApiError, request, tokens } from "@/api/client";
import { LoginPage } from "@/features/auth/LoginPage";
import { AuthProvider } from "@/lib/auth";

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("api client", () => {
  it("validates responses with zod", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json({ x: "not a number" }));
    await expect(request("/api/x/", z.object({ x: z.number() }))).rejects.toMatchObject({ code: "contract_violation" });
  });

  it("maps backend error envelopes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json({ error: "serving_unavailable", detail: "model server unavailable" }, 503));
    const err = await request("/api/x/", z.any()).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
    expect(err.message).toBe("model server unavailable");
  });

  it("refreshes the access token once on 401 and retries", async () => {
    tokens.set("old", "refresh-1");
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockImplementationOnce(() => json({ detail: "expired" }, 401))
      .mockImplementationOnce(() => json({ access: "new", refresh: "refresh-2" }))
      .mockImplementationOnce(() => json({ ok: true }));
    const res = await request("/api/x/", z.object({ ok: z.boolean() }));
    expect(res.ok).toBe(true);
    expect(tokens.access).toBe("new");
    const retried = fetchMock.mock.calls[2][1] as RequestInit;
    expect(new Headers(retried.headers).get("Authorization")).toBe("Bearer new");
  });
});

describe("login page", () => {
  it("logs in and redirects", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      json({ access: "a", refresh: "r", user: { id: 1, username: "officer", email: "", first_name: "", last_name: "", role: "officer" } }),
    );
    render(
      <QueryClientProvider client={new QueryClient()}>
        <AuthProvider>
          <MemoryRouter initialEntries={["/login"]}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/applications" element={<div>applications page</div>} />
            </Routes>
          </MemoryRouter>
        </AuthProvider>
      </QueryClientProvider>,
    );
    await userEvent.type(screen.getByLabelText("Username"), "officer");
    await userEvent.type(screen.getByLabelText("Password"), "secret");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.getByText("applications page")).toBeInTheDocument());
    expect(tokens.access).toBe("a");
  });

  it("shows the error on bad credentials", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => json({ detail: "No active account found" }, 401));
    render(
      <QueryClientProvider client={new QueryClient()}>
        <AuthProvider>
          <MemoryRouter initialEntries={["/login"]}>
            <LoginPage />
          </MemoryRouter>
        </AuthProvider>
      </QueryClientProvider>,
    );
    await userEvent.type(screen.getByLabelText("Username"), "x");
    await userEvent.type(screen.getByLabelText("Password"), "y");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No active account found");
  });
});
