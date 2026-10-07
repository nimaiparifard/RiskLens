{{- define "risklens.image" -}}
{{ .root.Values.image.registry }}/risklens-{{ .name }}:{{ .root.Values.image.tag }}
{{- end -}}

{{- define "risklens.labels" -}}
app.kubernetes.io/part-of: risklens
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "risklens.backendEnv" -}}
envFrom:
  - secretRef: { name: {{ .Values.existingSecret }} }
env:
  - { name: SERVING_URL, value: "http://{{ .Release.Name }}-serving:8001" }
{{- range $k, $v := .Values.env }}
  - { name: {{ $k }}, value: {{ $v | quote }} }
{{- end }}
{{- end -}}
