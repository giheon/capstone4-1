import { motion } from 'motion/react';
import { Brain, Check } from 'lucide-react';
import { useEffect, useState } from 'react';

interface SmartLoadingScreenProps {
  onComplete: () => void;
}

const steps = [
  "문제 조건 및 난이도 스캔 완료",
  "최적화된 AI 모델 매칭 중...",
  "실전 사고 개연성 검증 중..."
];

export function SmartLoadingScreen({ onComplete }: SmartLoadingScreenProps) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 1200);

    const completeTimeout = setTimeout(() => {
      onComplete();
    }, 4000);

    return () => {
      clearInterval(stepInterval);
      clearTimeout(completeTimeout);
    };
  }, [onComplete]);

  return (
    <div className="w-full h-full bg-white flex flex-col items-center justify-center p-4">
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="relative mb-8"
      >
        <div className="relative w-24 h-24">
          <motion.div
            className="absolute inset-0 rounded-full bg-gradient-to-br from-[#2563EB] to-[#60A5FA] opacity-20"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.2, 0.1, 0.2],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
            }}
          />

          <motion.div
            className="absolute inset-4 rounded-full bg-gradient-to-br from-[#2563EB] to-[#60A5FA] opacity-40"
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.4, 0.2, 0.4],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: 0.2,
            }}
          />

          <div className="absolute inset-6 rounded-full bg-gradient-to-br from-[#2563EB] to-[#60A5FA] flex items-center justify-center">
            <Brain className="w-8 h-8 text-white" />
          </div>
        </div>

        <motion.div
          className="absolute -top-1 -right-1 w-4 h-4 bg-[#2563EB] rounded-full"
          animate={{
            y: [-3, 3, -3],
            x: [3, -3, 3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
        />
        <motion.div
          className="absolute -bottom-1 -left-1 w-3 h-3 bg-[#60A5FA] rounded-full"
          animate={{
            y: [3, -3, 3],
            x: [-3, 3, -3],
          }}
          transition={{
            duration: 2.5,
            repeat: Infinity,
          }}
        />
      </motion.div>

      <motion.h1
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-lg text-[#1E293B] text-center mb-1"
      >
        최적의 풀이법을
      </motion.h1>
      <motion.h1
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-lg text-[#1E293B] text-center mb-8"
      >
        분석하고 있습니다...
      </motion.h1>

      <div className="w-full bg-white rounded-xl border border-gray-200 p-4 shadow-lg">
        <div className="space-y-3">
          {steps.map((step, index) => (
            <motion.div
              key={step}
              initial={{ x: -20, opacity: 0 }}
              animate={{
                x: 0,
                opacity: currentStep >= index ? 1 : 0.3,
              }}
              transition={{ delay: index * 0.3 }}
              className="flex items-start gap-2.5"
            >
              <div className={`mt-0.5 flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center ${
                currentStep >= index
                  ? 'bg-[#2563EB]'
                  : 'bg-gray-200'
              }`}>
                {currentStep >= index && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                  >
                    <Check className="w-2.5 h-2.5 text-white" />
                  </motion.div>
                )}
              </div>
              <p className={`text-xs ${
                currentStep >= index
                  ? 'text-[#1E293B]'
                  : 'text-gray-400'
              }`}>
                {currentStep === index && index === steps.length - 1 ? step : step.replace('중...', currentStep > index ? '완료' : '중...')}
              </p>
            </motion.div>
          ))}
        </div>
      </div>

      <motion.div
        className="mt-6 w-40 h-1 bg-gray-200 rounded-full overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <motion.div
          className="h-full bg-gradient-to-r from-[#2563EB] to-[#60A5FA]"
          initial={{ width: "0%" }}
          animate={{ width: "100%" }}
          transition={{ duration: 3.5, ease: "easeInOut" }}
        />
      </motion.div>
    </div>
  );
}
