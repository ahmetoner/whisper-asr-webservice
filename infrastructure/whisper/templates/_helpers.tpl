{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "whisper-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "whisper-api.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "whisper-api.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Generate basic labels */}}
{{- define "whisper-api.labels" -}}
labels:
  app.kubernetes.io/name: {{ include "whisper-api.name" . }}
  helm.sh/chart: {{ include "whisper-api.chart" . }}
  app.kubernetes.io/instance: {{ .Release.Name }}
  {{- if .Chart.AppVersion }}
  app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
  {{- end }}
  chart_version: {{ .Chart.Version }}
  app.kubernetes.io/managed-by: {{.Release.Service | quote }}
  amlg-product: whisper-api
  amlg-service: pplus
{{- end -}}
