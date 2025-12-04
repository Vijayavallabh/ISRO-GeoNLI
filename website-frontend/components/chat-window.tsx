"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Send, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Message {
  id: string
  role: "user" | "agent"
  text: string
  image?: string
}

interface ChatWindowProps {
  messages: Message[]
  onSendMessage: (prompt: string) => void
  isLoading: boolean
}

export default function ChatWindow({ messages, onSendMessage, isLoading }: ChatWindowProps) {
  const [input, setInput] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)

  const checkIfNearBottom = () => {
    if (!messagesContainerRef.current) return true
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current
    const threshold = 100 // pixels from bottom
    return scrollHeight - scrollTop - clientHeight < threshold
  }

  useEffect(() => {
    if (shouldAutoScroll || checkIfNearBottom()) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
      setShouldAutoScroll(true) // Reset after scrolling
    }
  }, [messages, isLoading])

  // Track scroll position to determine if user manually scrolled up
  const handleScroll = () => {
    setShouldAutoScroll(checkIfNearBottom())
  }

  const handleSend = () => {
    if (input.trim()) {
      onSendMessage(input)
      setInput("")
      setShouldAutoScroll(checkIfNearBottom())
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }
  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div
      ref={messageContainerRef}
      onScoll={handleScroll} 
      className="flex-1 overflow-y-auto p-6 border border-border rounded-lg m-4"
      >
        <div className="max-w-xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center">
              <div>
                <p className="text-muted-foreground">No messages yet</p>
                <p className="text-xs text-muted-foreground mt-2">Upload an image and ask questions about it</p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className="space-y-1">
                <div className="text-xs font-semibold text-muted-foreground uppercase">
                  {msg.role === "user" ? "You" : "Agent"}
                </div>
                <div
                  className={`p-3 rounded-lg ${msg.role === "user" ? "bg-blue-100 text-foreground" : "bg-muted text-foreground"}`}
                >
                  <p className="text-sm whitespace-pre-wrap break-words">{msg.text}</p>
                  {msg.image && (
                    <img
                      src={msg.image || "/placeholder.svg"}
                      alt="Response"
                      className="mt-2 rounded w-full max-w-xs"
                    />
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 size={16} className="animate-spin" />
              Agent is thinking...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe the image"
            className="flex-1 px-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            disabled={isLoading}
          />
          <Button onClick={handleSend} disabled={!input.trim() || isLoading} size="sm" className="gap-2">
            <Send size={16} />
            <span className="hidden sm:inline">Send</span>
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">⌘ Enter to send</p>
      </div>
    </div>
  )
}
