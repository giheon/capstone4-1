import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Lightbulb, CheckCircle2, ArrowLeft } from 'lucide-react';

interface ExplanationScreenProps {
  onReset: () => void;
}

interface TextStreamProps {
  text: string;
  delay?: number;
  onComplete?: () => void;
}

function TextStream({ text, delay = 0, onComplete }: TextStreamProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, delay + Math.random() * 20);

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && onComplete) {
      onComplete();
    }
  }, [currentIndex, text, delay, onComplete]);

  return <span>{displayedText}</span>;
}

export function ExplanationScreen({ onReset }: ExplanationScreenProps) {
  const [showKeyPoint, setShowKeyPoint] = useState(true);
  const [showStep1, setShowStep1] = useState(false);
  const [showStep1Detail, setShowStep1Detail] = useState(false);
  const [showStep2, setShowStep2] = useState(false);
  const [showStep2Detail, setShowStep2Detail] = useState(false);
  const [showStep3, setShowStep3] = useState(false);
  const [showStep3Detail, setShowStep3Detail] = useState(false);
  const [showAnswer, setShowAnswer] = useState(false);

  return (
    <div className="w-full h-full bg-blue-50 flex flex-col">
      <button onClick={onReset} className="absolute top-4 left-4 z-20 w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center hover:bg-gray-50 hover:scale-110 active:scale-95 transition-all duration-200">
        <ArrowLeft className="w-5 h-5 text-blue-600" strokeWidth={2.5} />
      </button>
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 pb-6 pt-16">
        {showKeyPoint && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="mb-4"
          >
            <div className="relative bg-blue-500 rounded-2xl p-4 overflow-hidden shadow-lg">
              <div className="relative z-10">
                <div className="flex items-start gap-2 mb-2">
                  <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center flex-shrink-0">
                    <Lightbulb className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-white text-sm mb-1">💡 실전 핵심 착안</h3>
                  </div>
                </div>
                <p className="text-white text-xs pl-10">
                  <TextStream
                    text="이 조건에서는 [함수의 대칭성]을 가장 먼저 의심해야 합니다."
                    delay={30}
                    onComplete={() => setTimeout(() => setShowStep1(true), 500)}
                  />
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {showStep1 && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="mb-4"
          >
            <h3 className="text-blue-900 text-sm mb-3 flex items-center gap-2">
              <span className="text-blue-600">개연성 흐름도</span>
            </h3>

            <div className="bg-white rounded-xl p-4 border border-blue-100 shadow-md">
              <div className="text-blue-900 text-xs leading-relaxed space-y-3">
                <div className="flex items-start gap-2">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs">
                    1
                  </div>
                  <p className="flex-1 pt-0.5">
                    <TextStream
                      text="조건 (가)에 의해 f(x+2) = f(2-x)가 성립합니다. 따라서 자연스럽게 x=2를 중심으로 대칭임을 알 수 있습니다."
                      delay={30}
                      onComplete={() => setTimeout(() => setShowStep2(true), 300)}
                    />
                  </p>
                </div>
                {showStep2 && (
                  <div className="flex items-start gap-2">
                    <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs">
                      2
                    </div>
                    <p className="flex-1 pt-0.5">
                      <TextStream
                        text="대칭성을 활용하면 좌극한과 우극한이 같습니다. 여기서 정석대로 풀면 오래 걸리므로, 직관적으로 대칭점의 함수값을 이용합니다."
                        delay={30}
                        onComplete={() => setTimeout(() => setShowStep3(true), 300)}
                      />
                    </p>
                  </div>
                )}
                {showStep3 && (
                  <div className="flex items-start gap-2">
                    <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs">
                      3
                    </div>
                    <p className="flex-1 pt-0.5">
                      <TextStream
                        text="조건 (나)를 대입하여 계산하면 f(2) = 3이고, 대칭성에 의해 극한값도 3입니다."
                        delay={30}
                        onComplete={() => setTimeout(() => setShowAnswer(true), 500)}
                      />
                    </p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {showAnswer && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="mb-5"
          >
            <div className="relative bg-blue-500 rounded-xl p-3 overflow-hidden shadow-lg">
              <div className="relative z-10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-blue-600 rounded-lg flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-white text-sm">최종 정답</h3>
                </div>
                <div className="text-2xl text-white">
                  ⑤
                </div>
              </div>
            </div>
          </motion.div>
        )}
        </div>
      </div>
    </div>
  );
}
