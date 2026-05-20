import { Camera } from 'lucide-react';

interface HomeScreenProps {
  onStartCamera: () => void;
}

export function HomeScreen({ onStartCamera }: HomeScreenProps) {
  return (
    <div className="w-full h-full bg-blue-50 flex flex-col">
      <div className="flex-1 flex items-center justify-center">
        <button
          onClick={onStartCamera}
          className="flex flex-col items-center gap-4 transform hover:scale-105 active:scale-95 transition-all duration-200"
        >
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center shadow-xl hover:shadow-2xl transition-all">
            <Camera className="w-10 h-10 text-white" strokeWidth={2.5} />
          </div>
          <div className="text-center">
            <p className="text-lg text-blue-900 mb-1" style={{ fontWeight: 600 }}>수능수학 AI</p>
            <p className="text-sm text-gray-500">문제를 촬영하면 AI가 실시간으로 해설합니다</p>
          </div>
        </button>
      </div>
    </div>
  );
}
