const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000/api"

export async function callChatAPI(
  sessionId: string,
  prompt: string,
  imageFile: File | null,
  queryType = "general",
  spatialResolution = 10,
  token: string,
) {
  const formData = new FormData()
  formData.append("session_id", sessionId)
  formData.append("query", prompt)
  formData.append("query_type", queryType)
  formData.append("spatial_resolution_m", spatialResolution.toString())

  if (imageFile) {
    formData.append("image", imageFile)
  }

  const response = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Chat API error: ${response.statusText}`)
  }

  return response.json()
}

export async function getChatHistory(token: string) {
  const response = await fetch(`${BACKEND_URL}/chat-history`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (!response.ok) {
    throw new Error(`Chat history API error: ${response.statusText}`)
  }

  return response.json()
}

export async function getSessionHistory(sessionId: string) {
  const response = await fetch(`${BACKEND_URL}/get-history?sessionId=${sessionId}`)

  if (!response.ok) {
    throw new Error(`Session history API error: ${response.statusText}`)
  }

  return response.json()
}

export async function login(email: string, password: string) {
  const response = await fetch(`${BACKEND_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  })

  if (!response.ok) {
    throw new Error(`Login error: ${response.statusText}`)
  }

  return response.json()
}
