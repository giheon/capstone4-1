import { ArrowLeft, TrendingUp, Award, Target } from 'lucide-react';
import { motion } from 'motion/react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';

interface AnalysisScreenProps {
  onBack: () => void;
}

const subjectData = [
  { name: '미적분', score: 85, color: '#2563EB' },
  { name: '확률통계', score: 72, color: '#8B5CF6' },
  { name: '기하', score: 65, color: '#EC4899' },
];

const weekData = [
  { day: '월', problems: 8 },
  { day: '화', problems: 12 },
  { day: '수', problems: 6 },
  { day: '목', problems: 15 },
  { day: '금', problems: 11 },
  { day: '토', problems: 9 },
  { day: '일', problems: 5 },
];

export function AnalysisScreen({ onBack }: AnalysisScreenProps) {
  return (
    <div className="w-full h-full bg-white flex flex-col">
      <div className="flex-shrink-0 bg-white px-4 py-3 border-b border-gray-100 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="w-9 h-9 rounded-full bg-gray-50 flex items-center justify-center hover:bg-gray-100">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <h1 className="text-lg text-[#1E293B]">성적 분석</h1>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="px-4 py-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-4 mb-4 border border-purple-100"
          >
            <div className="flex items-center gap-2 mb-3">
              <Award className="w-5 h-5 text-purple-600" />
              <span className="text-sm text-[#1E293B]">종합 성적</span>
            </div>
            <div className="text-center">
              <p className="text-4xl text-purple-600 mb-1">74.3</p>
              <p className="text-xs text-gray-600">평균 점수</p>
              <div className="mt-2 flex items-center justify-center gap-1">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className="text-xs text-green-600">지난주 대비 +5.2점</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-4"
          >
            <h2 className="text-sm text-[#1E293B] mb-3">과목별 성적</h2>
            <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
              <div className="space-y-4">
                {subjectData.map((subject, index) => (
                  <div key={subject.name}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-[#1E293B]">{subject.name}</span>
                      <span className="text-sm" style={{ color: subject.color }}>{subject.score}점</span>
                    </div>
                    <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${subject.score}%` }}
                        transition={{ delay: 0.2 + index * 0.1, duration: 0.6 }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: subject.color }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-4"
          >
            <h2 className="text-sm text-[#1E293B] mb-3">주간 학습량</h2>
            <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={weekData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                  <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
                  <Bar dataKey="problems" radius={[6, 6, 0, 0]}>
                    {weekData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill="#2563EB" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-sm text-[#1E293B] mb-3">개선 목표</h2>
            <div className="space-y-2">
              <div className="bg-white rounded-xl p-3 border border-gray-200 shadow-sm">
                <div className="flex items-start gap-2">
                  <Target className="w-5 h-5 text-[#2563EB] flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-[#1E293B] mb-1">기하 영역 집중 학습</p>
                    <p className="text-xs text-gray-600">공간벡터와 이차곡선 문제를 매일 3개씩 풀어보세요</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-xl p-3 border border-gray-200 shadow-sm">
                <div className="flex items-start gap-2">
                  <Target className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-[#1E293B] mb-1">풀이 시간 단축</p>
                    <p className="text-xs text-gray-600">평균 풀이 시간을 3분 이내로 줄여보세요</p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
