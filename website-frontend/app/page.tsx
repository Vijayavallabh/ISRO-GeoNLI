"use client"

import { useState, useRef, useEffect } from "react"
import { PencilIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import Sidebar from "@/components/sidebar"
import ChatWindow from "@/components/chat-window"
import ImagePanel from "@/components/image-panel"
import UploadDialog from "@/components/upload-dialog"
import { getChatHistory, getSessionHistory } from "@/lib/api-client"

interface Message {
  id: string
  role: "user" | "agent"
  text: string
  image?: string
}

interface ChatSession {
  sessionId: string
  title: string
  image?: string
  messages: Message[]
  imageFile?: File
  imageBase64?: string
  spatialResolution?: number
}

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentChatId, setCurrentChatId] = useState("")
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState("")
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [token, setToken] = useState<string | null>(null)
  const [isInitializing, setIsInitializing] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [spatialResolution, setSpatialResolution] = useState<number>(1.0)  // ADD THIS

  useEffect(() => {
    const storedToken = localStorage.getItem("token")
    if (storedToken) {
      setToken(storedToken)
      loadChatHistory(storedToken)
    } else {
      setIsInitializing(false)
      // Redirect to login if no token
    }
  }, [])

  const loadChatHistory = async (authToken: string) => {
    try {
      const history = await getChatHistory(authToken)
      if (history && Array.isArray(history)) {
        // Load sessions with their message history
        const sessionsWithHistory = await Promise.all(
          history.map(async (session: { sessionId: string; title: string }) => {
            try {
              const sessionData = await getSessionHistory(session.sessionId)
              return {
                sessionId: session.sessionId,
                title: session.title,
                image: sessionData.image?.image_url,
                messages: sessionData.messages.map((msg: any, idx: number) => ({
                  id: `${session.sessionId}-${idx}`,
                  role: msg.role === "assistant" ? "agent" : msg.role,
                  text: msg.content || "",
                  image: msg.type === "image" ? msg.content : undefined,
                })),
              }
            } catch (error) {
              console.log("[v0] Error loading session history:", error)
              return {
                sessionId: session.sessionId,
                title: session.title,
                messages: [],
              }
            }
          }),
        )
        setChatSessions(sessionsWithHistory)
        if (sessionsWithHistory.length > 0) {
          setCurrentChatId(sessionsWithHistory[0].sessionId)
        }
      }
    } catch (error) {
      console.log("[v0] Error loading chat history:", error)
    } finally {
      setIsInitializing(false)
    }
  }

  const currentChat = chatSessions.find((c) => c.sessionId === currentChatId)

  const handleSelectChat = (id: string) => {
    setCurrentChatId(id)
    setIsEditing(false)
  }

  const handleEditTitle = () => {
    if (currentChat) {
      setEditTitle(currentChat.title)
      setIsEditing(true)
    }
  }

  const handleSaveTitle = () => {
    if (editTitle.trim() && currentChat) {
      setChatSessions(chatSessions.map((c) => (c.sessionId === currentChatId ? { ...c, title: editTitle } : c)))
    }
    setIsEditing(false)
  }

  const handleImageUpload = (file: File) => {
    if (!currentChat) return

    if (currentChat.image && currentChat.messages.length > 0) {
      setShowUploadDialog(true)
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      const imageBase64 = e.target?.result as string
      setChatSessions(
        chatSessions.map((c) =>
          c.sessionId === currentChatId 
            ? { 
                ...c, 
                image: imageBase64, 
                imageFile: file, 
                imageBase64, 
                messages: [],
                spatialResolution: spatialResolution  // Preserve spatial resolution
              } 
            : c,
        ),
      )
    }
    reader.readAsDataURL(file)
  }

  const handleSendMessage = async (prompt: string) => {
    if (!currentChat || !currentChat.image || !token) return

    setIsLoading(true)

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      text: prompt,
    }

    // Update chat with user message
    setChatSessions(
      chatSessions.map((c) => (c.sessionId === currentChatId ? { ...c, messages: [...c.messages, userMessage] } : c)),
    )

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: currentChat.sessionId,
          prompt: prompt,
          imageBase64: currentChat.imageBase64,
          spatialResolution: currentChat.spatialResolution || spatialResolution,  // ADD THIS
          token: token,
        }),
      })


      const data = await response.json()

      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "agent",
        text: data.response || "",
        image: data.returned_image,
      }

      // Update chat with agent message
      setChatSessions(
        chatSessions.map((c) =>
          c.sessionId === currentChatId ? { ...c, messages: [...c.messages, agentMessage] } : c,
        ),
      )
    } catch (error) {
      console.log("[v0] Error sending message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        role: "agent",
        text: "Sorry, there was an error processing your request. Please try again.",
      }
      setChatSessions(
        chatSessions.map((c) =>
          c.sessionId === currentChatId ? { ...c, messages: [...c.messages, errorMessage] } : c,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }
  const handleSpatialResolutionChange = (value: number) => {
    setSpatialResolution(value)
    // Also update current chat session
    if (currentChat) {
      setChatSessions(
        chatSessions.map((c) =>
          c.sessionId === currentChatId ? { ...c, spatialResolution: value } : c
        )
      )
    }
  }


  const handleNewChat = () => {
    const newId = Date.now().toString()
    const newChat: ChatSession = {
      sessionId: newId,
      title: "New Chat",
      messages: [],
    }
    setChatSessions([newChat, ...chatSessions])
    setCurrentChatId(newId)
  }

  const handleStartNewChatWithImage = () => {
    handleNewChat()
    setShowUploadDialog(false)
    setTimeout(() => {
      fileInputRef.current?.click()
    }, 0)
  }

  if (isInitializing) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>
  }

  if (!token) {
    // Redirect to login page instead of just showing message
    if (typeof window !== 'undefined') {
      window.location.href = '/login'
    }
    return <div className="flex items-center justify-center h-screen">Redirecting to login...</div>
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        chatSessions={chatSessions}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-border px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 flex-1">
            {isEditing ? (
              <input
                type="text"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveTitle()
                  if (e.key === "Escape") setIsEditing(false)
                }}
                className="text-lg font-medium outline-none border-b-2 border-ring px-1 flex-1 bg-background"
                autoFocus
              />
            ) : (
              <>
                <h1 className="text-lg font-medium">{currentChat?.title}</h1>
                <button onClick={handleEditTitle} className="text-muted-foreground hover:text-foreground ml-2">
                  <PencilIcon size={16} />
                </button>
              </>
            )}
          </div>
          {isEditing && (
            <Button onClick={handleSaveTitle} variant="ghost" size="sm" className="ml-2">
              Save
            </Button>
          )}
        </div>

        {/* Content Area - Split Layout */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left: Chat Window */}
          <div className="flex-1 border-r border-border flex flex-col">
            <ChatWindow
              messages={currentChat?.messages || []}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          </div>

          {/* Right: Image Panel */}
          <div className="flex-1 flex flex-col">
            <ImagePanel 
              image={currentChat?.image} 
              onUpload={() => fileInputRef.current?.click()}
              spatialResolution={currentChat?.spatialResolution || spatialResolution}
              onSpatialResolutionChange={handleSpatialResolutionChange}
            />
          </div>
        </div>
      </main>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) {
            handleImageUpload(file)
          }
        }}
      />

      {/* Upload Conflict Dialog */}
      <UploadDialog
        isOpen={showUploadDialog}
        onClose={() => setShowUploadDialog(false)}
        onConfirm={handleStartNewChatWithImage}
      />
    </div>
  )
}
