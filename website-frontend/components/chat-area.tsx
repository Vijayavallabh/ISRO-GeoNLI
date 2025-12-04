"use client"

interface ChatAreaProps {
  image: string
}

export default function ChatArea({ image }: ChatAreaProps) {
  return (
    <div className="flex-1 overflow-y-auto px-6 py-6">
      <div className="flex justify-center">
        <div className="w-full max-w-2xl">
          <img
            src={image || "/placeholder.svg"}
            alt="Chat image"
            className="w-full rounded-2xl shadow-sm border border-gray-200 object-cover"
            style={{ maxHeight: "500px" }}
          />
        </div>
      </div>
    </div>
  )
}
