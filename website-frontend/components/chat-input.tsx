"use client"

import type React from "react"

import { useState } from "react"
import { ArrowRight } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

interface ChatInputProps {
  onAddChat: (title: string) => void
}

export default function ChatInput({ onAddChat }: ChatInputProps) {
  const [input, setInput] = useState("")

  const handleSubmit = () => {
    if (input.trim()) {
      onAddChat(input)
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-gray-200 px-6 py-4">
      <div className="flex gap-2 max-w-2xl mx-auto">
        <Input
          placeholder="Describe the image"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-white border-gray-300 text-gray-900 placeholder:text-gray-400"
        />
        <Button onClick={handleSubmit} variant="ghost" size="sm" className="px-3" disabled={!input.trim()}>
          <ArrowRight size={20} className="text-gray-600" />
        </Button>
        <div className="text-xs text-gray-400 flex items-center">cmd⏎</div>
      </div>
    </div>
  )
}
