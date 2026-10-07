import { Navigate, type RouteObject } from "react-router-dom";

import { ProtectedLayout } from "@/components/Layout";
import { ApplicationDetailPage } from "@/features/applications/ApplicationDetailPage";
import { ApplicationsPage } from "@/features/applications/ApplicationsPage";
import { NewApplicationPage } from "@/features/applications/NewApplicationPage";
import { LoginPage } from "@/features/auth/LoginPage";
import { FraudMonitorPage } from "@/features/fraud/FraudMonitorPage";
import { ModelsPage } from "@/features/models/ModelsPage";
import { MonitoringPage } from "@/features/monitoring/MonitoringPage";

export const routes: RouteObject[] = [
  { path: "/login", element: <LoginPage /> },
  {
    element: <ProtectedLayout />,
    children: [
      { path: "/", element: <Navigate to="/applications" replace /> },
      { path: "/applications", element: <ApplicationsPage /> },
      { path: "/applications/new", element: <NewApplicationPage /> },
      { path: "/applications/:id", element: <ApplicationDetailPage /> },
      { path: "/fraud", element: <FraudMonitorPage /> },
      { path: "/models", element: <ModelsPage /> },
      { path: "/monitoring", element: <MonitoringPage /> },
    ],
  },
  { path: "*", element: <Navigate to="/" replace /> },
];
