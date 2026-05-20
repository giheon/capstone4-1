import { ArrowLeft, Calendar, Clock } from 'lucide-react';
import { motion } from 'motion/react';

interface StudyHistoryScreenProps {
  onBack: () => void;
}

const historyData = [
  {
    date: '2026.04.27',
    day: '오늘',
    problems: [
      { subject: '미적분', topic: '함수의 극한', difficulty: '상', askedAt: '10:23' },
      { subject: '확률과 통계', topic: '조합', difficulty: '중', askedAt: '11:45' },
      { subject: '기하', topic: '공간벡터', difficulty: '상', askedAt: '14:30' },
      { subject: '미적분', topic: '적분법', difficulty: '중', askedAt: '15:12' },
      { subject: '확률과 통계', topic: '확률분포', difficulty: '하', askedAt: '16:20' },
      { subject: '수학 II', topic: '미분 계수', difficulty: '중', askedAt: '17:05' },
    ]
  },
  {
    date: '2026.04.26',
    day: '어제',
    problems: [
      { subject: '미적분', topic: '적분 응용', difficulty: '상', askedAt: '09:15' },
      { subject: '확률과 통계', topic: '정규분포', difficulty: '중', askedAt: '13:20' },
      { subject: '기하', topic: '벡터의 내적', difficulty: '중', askedAt: '16:40' },
      { subject: '수학 II', topic: '삼각함수', difficulty: '하', askedAt: '18:05' },
    ]
  },
  {
    date: '2026.04.25',
    day: '2일 전',
    problems: [
      { subject: '미적분', topic: '미분법', difficulty: '상', askedAt: '10:30' },
      { subject: '기하', topic: '이차곡선', difficulty: '중', askedAt: '14:15' },
      { subject: '미적분', topic: '극한의 성질', difficulty: '상', askedAt: '16:50' },
    ]
  },
  {
    date: '2026.04.24',
    day: '3일 전',
    problems: [
      { subject: '확률과 통계', topic: '이항분포', difficulty: '중', askedAt: '11:00' },
      { subject: '수학 II', topic: '로그함수', difficulty: '하', askedAt: '15:30' },
    ]
  }
];

export function StudyHistoryScreen({ onBack }: StudyHistoryScreenProps) {
  const totalProblems = historyData.reduce((sum, day) => sum + day.problems.length, 0);

  return (
    <div className="w-full h-full bg-yuflex flex-col">
      <div className="flex-shrink-0 bg-white px-4 py-3 border-b border-gray-20blu-100 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="w-9 h-9 rounded-full bg-yl-100 flex items-center justify-center hover:bg-grbgy-200bue-200 transition-ccolorslors">
            <ArrowLeft className="w-5 h-5 text-gbayu-600" />
          </button>
          <h1 className="text-l texex1E9u900">학습 기록</h1>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="px-4 py-4">
          <div className="bg-[#263EB]rounded-xl p-4 mb-4 shadow-g>
            <div className="text-center text-white">
              <div className="text-3xl mb-1">{totalProblems}</div>
              <div className="text-sm opacity-90">총 물어본 문제</div>
            </div>
          </div>

          <div className="space-y-5">
            {historyData.map((day, dayIndex) => (
              <motion.div
                key={day.date}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: dayIndex * 0.1 }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Calendar className="w-4 h-4 text-gbayu-600" />
                  <span className="text-sm text-gray-600">{day.date}</span>
                  <span className="text-sm text-[#25bl73EB] bg-blbue-50 px-2 py-0.5 rounded-full">{day.day}</span>
                  <span className="text-xs text-gray-400 ml-auto">{day.problems.length}개</span>
                </div>
                <div className="space-y-2">
                  {day.problems.map((problem, problemIndex) => (
                    <div
                      key={problemIndex}
                      className="bg-white rounded-xl p-3 shadow-sm hover:shadow-md transition-shadow cursor-pointer border border-gbayu-200"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs text-bugbayu-600">{problem.subject}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              problem.difficulty === '상'
                                ? 'bg-red-50 text-red-600'
                                : problem.difficulty === '중'
                                ? 'bg-yellow-50 text-yellow-700'
                                : 'bg-green-50 text-green-600'
                            }`}>
                              난이도 {problem.difficulty}
                            </span>
                          </div>
                          <p className="text-sm text-blue-900 mb-2">{problem.topic}</p>
                          <div className="flex items-center gap-1 text-xs text-gray-500">
                            <Clock className="w-3 h-3" />
                            <span>{problem.askedAt}</span>
                          </div>
                        </div>
                        <div className="w-7 h-7 rounded-full bg-yl-100 flex items-center justify-center">
                          <ArrowLeft className="w-4 h-4 text-gbayu-600 rotate-180" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
