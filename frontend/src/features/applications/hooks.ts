import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type ApplicationFilters } from "@/api/endpoints";

export const useApplications = (filters: ApplicationFilters) =>
  useQuery({ queryKey: ["applications", filters], queryFn: () => api.applications(filters), placeholderData: keepPreviousData });

export const useApplication = (id: string) =>
  useQuery({ queryKey: ["application", id], queryFn: () => api.application(id) });

export const useAudit = (id: string) => useQuery({ queryKey: ["audit", id], queryFn: () => api.audit(id) });

export const useStats = () => useQuery({ queryKey: ["stats"], queryFn: api.stats });

function useInvalidateApplication(id: string) {
  const qc = useQueryClient();
  return () => {
    qc.invalidateQueries({ queryKey: ["application", id] });
    qc.invalidateQueries({ queryKey: ["audit", id] });
    qc.invalidateQueries({ queryKey: ["applications"] });
    qc.invalidateQueries({ queryKey: ["stats"] });
  };
}

export function useScore(id: string) {
  const invalidate = useInvalidateApplication(id);
  return useMutation({ mutationFn: () => api.scoreApplication(id), onSuccess: invalidate });
}

export function useDecide(id: string) {
  const invalidate = useInvalidateApplication(id);
  return useMutation({
    mutationFn: (v: { decision: "approve" | "reject"; comment: string }) => api.decide(id, v.decision, v.comment),
    onSuccess: invalidate,
  });
}
