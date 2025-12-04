"use client"

import { AlertCircle } from "lucide-react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

interface UploadDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
}

export default function UploadDialog({ isOpen, onClose, onConfirm }: UploadDialogProps) {
  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="text-destructive" size={20} />
            <AlertDialogTitle>Start New Chat?</AlertDialogTitle>
          </div>
        </AlertDialogHeader>
        <AlertDialogDescription>
          You already have questions asked about the current image. Uploading a new image will create a new chat
          session. Do you want to continue?
        </AlertDialogDescription>
        <div className="flex gap-3 justify-end">
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>Create New Chat</AlertDialogAction>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  )
}
