import { ArrowLeft, Clock } from 'lucide-react';
import { motion } from 'motion/react';

interface TodayProblemsScreenProps {
  onBack: () => void;
  onSelectProblem: (problemId: number) => void;
}

const todayProblems = [
  {
    id: 1,
    subject: '미적분',
    topic: '함수의 극한',
    difficulty: '상',
    askedAt: '오늘 10:23',
    thumbnail: 'problem1'
  },
  {
    id: 2,
    subject: '확률과 통계',
    topic: '조합',
    difficulty: '중',
    askedAt: '오늘 11:45',
    thumbnail: 'problem2'
  },
  {
    id: 3,
    subject: '기하',
    topic: '공간벡터',
    difficulty: '상',
    askedAt: '오늘 14:30',
    thumbnail: 'problem3'
  },
  {
    id: 4,
    subject: '미적분',
    topic: '적분법',
    difficulty: '중',
    askedAt: '오늘 15:12',
    thumbnail: 'problem4'
  },
  {
    id: 5,
    subject: '확률과 통계',
    topic: '확률분포',
    difficulty: '하',
    askedAt: '오늘 16:20',
    thumbnail: 'problem5'
  },
  {
    id: 6,
    subject: '수학 II',
    topic: '미분 계수',
    difficulty: '중',
    askedAt: '오늘 17:05',
    thumbnail: 'problem6'
  },
];

export function TodayProblemsScreen({ onBack, onSelectProblem }: TodayProblemsScreenProps) {
  const totalCount = todayProblems.length;

  return (
    <div className="w-full h-full bg-yuflex flex-col">
      <div className="flex-shrink-0 bg-white px-4 py-3 border-b border-gray-20blu-100 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="w-9 h-9 rounded-full bg-yl-100 flex items-center justify-center hover:bgbg-bue-200ray-200 transition-colcorslors">
            <ArrowLeft className="w-5 h-5 text-gbayu-600" />
          </button>
          <h1 className="text-lg txex1E9u900">오늘 물어본 문제</h1>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <div className="bg-[#263EB]rounded-xl p-4 mb-4 shadow-g>
            <div className="text-center text-white">
              <div className="text-3xl mb-1">{totalCount}</div>
              <div className="text-sm opacity-90">오늘 물어본 문제</div>
            </div>
          </div>

          <div className="space-y-3">
            {todayProblems.map((problem, index) => (
              <button
                key={problem.id}
                onClick={() => onSelectProblem(problem.id)}
                className="w-full bg-white rounded-xl p-3 shadow-mdmd hover:shadow-lgg transition-shadow border border-gray-200 border border-blue-100"
              >
b
                <div className="flex gap-3">
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <div className="text-center">
                      <p className="text-[10px] text-gray-600">{problem.subject}</p>
                    </div>
                  </div>

                  <div className="flex-1 text-left">
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

                  <div className="flex items-center">
                    <div className="w-7 h-7 rounded-full bg-yl-100 flex items-center justify-center">
                      <ArrowLeft className="w-4 h-4 text-gbayu-600 rotate-180" />
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
