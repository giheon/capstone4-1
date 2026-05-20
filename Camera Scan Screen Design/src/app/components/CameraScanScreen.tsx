import { ArrowLeft } from 'lucide-react';
import { motion } from 'motion/react';

interface CameraScanScreenProps {
  onCapture: () => void;
  onBack: () => void;
}

export function CameraScanScreen({ onCapture, onBack }: CameraScanScreenProps) {
  return (
    <div className="relative w-full h-full bg-blue-50 overflow-hidden">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDM3LDk5LDIzNSwwLjA1KSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-40" />

      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4">
        <button
          onClick={onBack}
          className="w-12 h-12 rounded-full bg-white shadow-md flex items-center justify-center text-blue-600 hover:bg-gray-50 hover:scale-110 active:scale-95 transition-all duration-200"
        >
          <ArrowLeft className="w-6 h-6" strokeWidth={2.5} />
        </button>
      </div>

      <div className="absolute inset-0 flex flex-col items-center justify-center px-4 pt-12 pb-40">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-blue-900 text-center mb-6"
        >
          <p className="text-base" style={{ fontWeight: 500 }}>풀고 싶은 수능 수학 문제를</p>
          <p className="text-base" style={{ fontWeight: 500 }}>영역에 맞춰주세요.</p>
        </motion.div>

        <div className="relative w-[85%] aspect-[2/3] rounded-2xl border-3 border-blue-500 bg-gradient-to-br from-gray-100 to-gray-200 shadow-2xl shadow-blue-500/20 overflow-hidden">
          <div className="absolute inset-0 border border-gray-300/30 rounded-xl m-1.5" />

          <div className="absolute top-3 left-3 w-6 h-6 border-t-3 border-l-3 border-blue-500 rounded-tl-lg" />
          <div className="absolute top-3 right-3 w-6 h-6 border-t-3 border-r-3 border-blue-500 rounded-tr-lg" />
          <div className="absolute bottom-3 left-3 w-6 h-6 border-b-3 border-l-3 border-blue-500 rounded-bl-lg" />
          <div className="absolute bottom-3 right-3 w-6 h-6 border-b-3 border-r-3 border-blue-500 rounded-br-lg" />

          <motion.div
            className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-blue-500 to-transparent shadow-lg shadow-blue-500/50"
            animate={{
              y: [0, 300, 0],
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 z-10 pb-10 flex justify-center">
        <motion.button
          whileTap={{ scale: 0.95 }}
          whileHover={{ scale: 1.05 }}
          onClick={onCapture}
          className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 shadow-2xl shadow-blue-500/50 flex items-center justify-center relative transition-all duration-200"
        >
          <div className="w-[70px] h-[70px] rounded-full border-4 border-white" />
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-white/50"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.5, 0, 0.5],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
            }}
          />
        </motion.button>
      </div>
    </div>
  );
}
