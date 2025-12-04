import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000/api"

interface Message {
  role: "user" | "agent"
  text: string
  image?: string
}

export async function POST(request: NextRequest) {
  try {
    const { sessionId, prompt, imageBase64, spatialResolution, token } = await request.json()

    if (!prompt || !sessionId) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    // Convert base64 image to File for FormData
    let imageFile: File | null = null
    if (imageBase64) {
      const byteString = atob(imageBase64.split(",")[1])
      const ab = new ArrayBuffer(byteString.length)
      const view = new Uint8Array(ab)
      for (let i = 0; i < byteString.length; i++) {
        view[i] = byteString.charCodeAt(i)
      }
      imageFile = new File([ab], "image.jpg", { type: "image/jpeg" })
    }

    // Prepare FormData for FastAPI backend
    const formData = new FormData()
    formData.append("session_id", sessionId)
    formData.append("query", prompt)
    formData.append("spatial_resolution_m", String(spatialResolution || 1.0))  // USE FROM REQUEST
    if (imageFile) {
      formData.append("image", imageFile)
    }

    // Call FastAPI backend
    const backendResponse = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend error: ${backendResponse.statusText}`)
    }

    const backendData = await backendResponse.json()

    // Transform response to match frontend expectations
    return NextResponse.json({
      response: backendData.reply || "",
      returned_image: backendData.image || undefined,
    })
  } catch (error) {
    console.log("[v0] Error in chat API:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
