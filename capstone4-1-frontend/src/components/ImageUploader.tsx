import React, { useState, useRef } from 'react';
import { Upload, Camera, X } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
  uploadedImage: string | null;
  onRemoveImage: () => void;
}

export function ImageUploader({ onImageUpload, uploadedImage, onRemoveImage }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageUpload(file);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('image/')) {
        onImageUpload(file);
      }
    }
  };

  const openFileSelector = () => {
    fileInputRef.current?.click();
  };

  if (uploadedImage) {
    return (
      <Card className="relative">
        <CardContent className="p-4">
          <div className="relative">
            <img 
              src={uploadedImage} 
              alt="업로드된 수학 문제" 
              className="w-full max-h-96 object-contain rounded-lg"
            />
            <Button
              variant="destructive"
              size="sm"
              className="absolute top-2 right-2"
              onClick={onRemoveImage}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`border-2 border-dashed transition-colors ${
      dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
    }`}>
      <CardContent className="p-8">
        <div
          className="flex flex-col items-center justify-center space-y-4 text-center"
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="p-4 rounded-full bg-muted">
            <Upload className="h-8 w-8 text-muted-foreground" />
          </div>
          
          <div className="space-y-2">
            <h3>수학 문제 이미지를 업로드하세요</h3>
            <p className="text-muted-foreground">
              드래그 앤 드롭하거나 클릭하여 파일을 선택하세요
            </p>
          </div>

          <div className="flex gap-2">
            <Button onClick={openFileSelector} className="gap-2">
              <Camera className="h-4 w-4" />
              파일 선택
            </Button>
          </div>

          <p className="text-xs text-muted-foreground">
            JPG, PNG, GIF 파일만 지원됩니다 (최대 10MB)
          </p>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
      </CardContent>
    </Card>
  );
}