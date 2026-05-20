import { ArrowLeft, MessageCircle, ThumbsUp, Eye, TrendingUp } from 'lucide-react';
import { motion } from 'motion/react';

interface CommunityScreenProps {
  onBack: () => void;
}

const posts = [
  {
    id: 1,
    tag: '공부법',
    title: '수능 2달 남았는데 미적분 어떻게 정리하나요?',
    author: '수험생A',
    time: '5분 전',
    likes: 23,
    comments: 12,
    views: 145,
    isHot: true,
  },
  {
    id: 2,
    tag: '질문',
    title: '확률과 통계 조합 문제 이해가 안 돼요 ㅠㅠ',
    author: '고3학생',
    time: '12분 전',
    likes: 8,
    comments: 5,
    views: 67,
    isHot: false,
  },
  {
    id: 3,
    tag: '풀이공유',
    title: '2026 9월 모평 22번 직관적 풀이법 공유',
    author: '수학고수',
    time: '1시간 전',
    likes: 156,
    comments: 34,
    views: 892,
    isHot: true,
  },
  {
    id: 4,
    tag: '정보',
    title: '킬러문항 대비 꿀팁 정리해봤습니다',
    author: '재수생',
    time: '2시간 전',
    likes: 89,
    comments: 21,
    views: 456,
    isHot: true,
  },
  {
    id: 5,
    tag: '응원',
    title: '오늘도 힘내요 여러분! 우리 모두 화이팅!',
    author: '같이공부',
    time: '3시간 전',
    likes: 42,
    comments: 18,
    views: 203,
    isHot: false,
  },
];

export function CommunityScreen({ onBack }: CommunityScreenProps) {
  return (
    <div className="w-full h-full bg-white flex flex-col">
      <div className="flex-shrink-0 bg-white px-4 py-3 border-b border-gray-100 z-10">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="w-9 h-9 rounded-full bg-gray-50 flex items-center justify-center hover:bg-gray-100">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <h1 className="text-lg text-[#1E293B]">커뮤니티</h1>
        </div>
      </div>

      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-100 z-10">
        <div className="flex gap-2 overflow-x-auto">
          <button className="px-3 py-1.5 bg-[#2563EB] text-white rounded-full text-xs whitespace-nowrap">
            전체
          </button>
          <button className="px-3 py-1.5 bg-gray-100 text-gray-600 rounded-full text-xs whitespace-nowrap">
            공부법
          </button>
          <button className="px-3 py-1.5 bg-gray-100 text-gray-600 rounded-full text-xs whitespace-nowrap">
            질문
          </button>
          <button className="px-3 py-1.5 bg-gray-100 text-gray-600 rounded-full text-xs whitespace-nowrap">
            풀이공유
          </button>
          <button className="px-3 py-1.5 bg-gray-100 text-gray-600 rounded-full text-xs whitespace-nowrap">
            응원
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto relative">
        <div className="px-4 py-3">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl p-3 mb-4 border border-amber-200"
          >
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-orange-600" />
              <span className="text-xs text-orange-700">실시간 인기 토픽</span>
            </div>
            <p className="text-sm text-[#1E293B]">#수능D-60 #미적분정복 #확통꿀팁</p>
          </motion.div>

          <div className="space-y-2">
            {posts.map((post, index) => (
              <motion.div
                key={post.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-white rounded-xl p-3 border border-gray-200 shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      post.tag === '공부법' ? 'bg-blue-100 text-blue-600' :
                      post.tag === '질문' ? 'bg-purple-100 text-purple-600' :
                      post.tag === '풀이공유' ? 'bg-green-100 text-green-600' :
                      post.tag === '정보' ? 'bg-orange-100 text-orange-600' :
                      'bg-pink-100 text-pink-600'
                    }`}>
                      {post.tag}
                    </span>
                    {post.isHot && (
                      <span className="px-1.5 py-0.5 bg-red-500 text-white rounded text-xs">HOT</span>
                    )}
                  </div>
                </div>

                <h3 className="text-sm text-[#1E293B] mb-2 line-clamp-1">{post.title}</h3>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>{post.author}</span>
                    <span>·</span>
                    <span>{post.time}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <ThumbsUp className="w-3 h-3" />
                      <span>{post.likes}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MessageCircle className="w-3 h-3" />
                      <span>{post.comments}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-3 h-3" />
                      <span>{post.views}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      <div className="absolute bottom-20 right-4">
        <button className="w-12 h-12 bg-[#2563EB] rounded-full shadow-lg shadow-blue-500/40 flex items-center justify-center text-white">
          <span className="text-xl">+</span>
        </button>
      </div>
    </div>
  );
}
