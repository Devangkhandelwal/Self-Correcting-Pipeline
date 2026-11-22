/**
 * Shared API client.
 *
 * The components previously called fetch() directly and funnelled every
 * failure into console.error without checking res.ok. With the backend down
 * or returning 500, the spinner simply stopped and the panel rendered blank
 * with no explanation. This surfaces a usable message instead.
 */

const BASE = '/api';

/** Error carrying the HTTP status and the server's own message, when present. */
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError(
      'Cannot reach the API. Is the Flask server running on port 5000?',
      0,
    );
  }

  let body = null;
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  if (isJson) {
    try {
      body = await response.json();
    } catch {
      body = null;
    }
  }

  if (!response.ok) {
    throw new ApiError(
      body?.error || `Request failed with HTTP ${response.status}.`,
      response.status,
    );
  }
  if (body === null) {
    throw new ApiError('The API returned a response that was not JSON.', response.status);
  }
  return body;
}

export const getHealth = (signal) => request('/health', { signal });

export const getPipelines = (signal) => request('/pipelines', { signal });

export const getAnomalyClasses = (signal) => request('/anomaly-classes', { signal });

export const runPipeline = (pipelineId, anomalyType, signal) =>
  request('/run-pipeline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pipeline_id: pipelineId, anomaly_type: anomalyType }),
    signal,
  });

export const getEvaluation = (signal) => request('/evaluate', { signal });

export const getAuditLogs = (limit = 30, signal) =>
  request(`/audit-logs?limit=${encodeURIComponent(limit)}`, { signal });
