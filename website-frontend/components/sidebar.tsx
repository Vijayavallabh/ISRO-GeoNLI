"use client"

import { Clock, Settings, Plus, ChevronLeft, ChevronRight } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import SettingsDialog from "@/components/settings-dialog"
import { useState } from "react"

interface ChatSession {
  id: string
  title: string
  image?: string
  messages: string[]
}

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  chatSessions: ChatSession[]
  currentChatId: string
  onSelectChat: (id: string) => void
  onNewChat: () => void
}

export default function Sidebar({
  isOpen,
  onToggle,
  chatSessions,
  currentChatId,
  onSelectChat,
  onNewChat,
}: SidebarProps) {
  const [showSettings, setShowSettings] = useState(false)
  const { theme, setTheme } = useTheme()

  return (
    <>
      <aside
        className={`bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-in-out flex flex-col ${
          isOpen ? "w-64" : "w-16"
        }`}
      >
        {/* Logo */}
        <div
          className={`px-4 py-6 border-b border-sidebar-border flex items-center justify-between ${
            !isOpen && "justify-center"
          }`}
        >
          {isOpen && (
            <div>
              <h1 className="text-base font-bold text-sidebar-foreground">Pruthvi</h1>
              <p className="text-xs text-sidebar-foreground/60">GeoNLI</p>
            </div>
          )}
          <button onClick={onToggle} className="text-sidebar-foreground hover:opacity-70 transition-opacity">
            {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </button>
        </div>

        {/* New Chat Button */}
        <div className={`px-3 py-4 border-b border-sidebar-border ${!isOpen && "px-2"}`}>
          <Button
            onClick={onNewChat}
            variant="outline"
            size="sm"
            className={`w-full gap-2 text-sidebar-foreground border-sidebar-border bg-sidebar-accent/50 hover:bg-sidebar-accent ${
              !isOpen && "px-2"
            }`}
          >
            <Plus size={16} />
            {isOpen && "New Chat"}
          </Button>
        </div>

        {/* Chat History */}
        {isOpen && (
          <div className="flex-1 overflow-y-auto px-3 py-4">
            <div className="flex items-center gap-2 mb-3">
              <Clock size={16} className="text-sidebar-foreground/60" />
              <h2 className="text-sm font-medium text-sidebar-foreground">Chat History</h2>
            </div>
            <div className="space-y-1">
              {chatSessions.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => onSelectChat(chat.id)}
                  className={`w-full text-left px-3 py-2 rounded text-sm truncate transition-colors ${
                    currentChatId === chat.id
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent/30"
                  }`}
                >
                  {chat.title}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Settings & User */}
        <div className={`border-t border-sidebar-border px-3 py-4 space-y-2 ${!isOpen && "px-2"}`}>
          <button
            onClick={() => setShowSettings(true)}
            className={`flex items-center gap-2 cursor-pointer hover:opacity-75 transition-opacity w-full ${
              !isOpen && "justify-center"
            }`}
          >
            <Settings size={16} className="text-sidebar-foreground" />
            {isOpen && <span className="text-sm text-sidebar-foreground">Settings</span>}
          </button>
          {isOpen && (
            <div className="flex items-center gap-2 px-3 py-2 rounded bg-sidebar-accent/50 cursor-pointer">
              <div className="w-6 h-6 rounded-full bg-sidebar-primary"></div>
              <span className="text-xs text-sidebar-foreground/60 truncate">user@example.com</span>
            </div>
          )}
        </div>
      </aside>

      {/* Settings Dialog */}
      <SettingsDialog
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        theme={theme}
        onThemeChange={setTheme}
      />
    </>
  )
}
