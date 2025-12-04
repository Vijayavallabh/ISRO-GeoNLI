"use client"

import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface ImagePanelProps {
  image?: string
  onUpload: () => void
  spatialResolution?: number
  onSpatialResolutionChange?: (value: number) => void
}

export default function ImagePanel({ 
  image, 
  onUpload, 
  spatialResolution = 1, /* Is there a way to keep this compulsory?*/
  onSpatialResolutionChange 
}: ImagePanelProps) {
  return (
    <div className="flex flex-col items-center justify-center p-6 border border-border rounded-lg m-4 h-full bg-muted/30">
      {image ? (
        <div className="w-full h-full flex flex-col items-center justify-center gap-4">
          <img src={image || "/placeholder.svg"} alt="Chat image" className="w-full h-full object-contain rounded-lg" />
          
          {/* Spatial Resolution Input */}
          <div className="w-full max-w-xs space-y-2">
            <label htmlFor="spatial-resolution" className="text-sm font-medium block mb-1">
              Spatial Resolution (m/pixel)
            </label>
            <Input
              id="spatial-resolution"
              type="number"
              step="0.01"
              min="0.01"
              value={spatialResolution}
              onChange={(e) => {
                const value = parseFloat(e.target.value) || 1.0
                onSpatialResolutionChange?.(value)
              }}
              placeholder="1.0"
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Ground sample distance in meters per pixel
            </p>
          </div>
          
          <Button onClick={onUpload} variant="outline" size="sm" className="mt-2 gap-2 bg-transparent">
            <Upload size={16} />
            Upload New Image
          </Button>
        </div>
      ) : (
        <div className="text-center">
          <div className="mb-4 p-4 bg-background rounded-lg">
            <Upload size={32} className="text-muted-foreground mx-auto" />
          </div>
          <p className="text-foreground font-medium mb-2">Upload an Image</p>
          <p className="text-xs text-muted-foreground mb-4">Start by uploading an image to analyze</p>
          <Button onClick={onUpload} className="gap-2">
            <Upload size={16} />
            Choose Image
          </Button>
        </div>
      )}
    </div>
  )
}