import React from 'react';
import { Clock, FileText, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

interface HistoryItem {
  id: string;
  timestamp: Date;
  problemType: string;
  score: number;
  thumbnail?: string;
}

interface AnalysisHistoryProps {
  history: HistoryItem[];
  onViewDetails: (id: string) => void;
}

export function AnalysisHistory({ history, onViewDetails }: AnalysisHistoryProps) {
  const getScoreBadgeVariant = (score: number) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  const averageScore = history.length > 0 
    ? Math.round(history.reduce((sum, item) => sum + item.score, 0) / history.length)
    : 0;

  return (
    <div className="space-y-6">
      {/* 통계 요약 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">총 문제 수</p>
                <p className="text-xl font-medium">{history.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg dark:bg-green-900">
                <TrendingUp className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">평균 점수</p>
                <p className="text-xl font-medium">{averageScore}점</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg dark:bg-blue-900">
                <Clock className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">이번 주</p>
                <p className="text-xl font-medium">
                  {history.filter(item => {
                    const weekAgo = new Date();
                    weekAgo.setDate(weekAgo.getDate() - 7);
                    return item.timestamp > weekAgo;
                  }).length}문제
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 히스토리 목록 */}
      <Card>
        <CardHeader>
          <CardTitle>분석 히스토리</CardTitle>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-center py-8">
              <div className="p-4 bg-muted rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <FileText className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="text-muted-foreground">아직 분석한 문제가 없습니다.</p>
              <p className="text-sm text-muted-foreground mt-1">첫 번째 수학 문제를 업로드해보세요!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4">
                    {item.thumbnail && (
                      <img 
                        src={item.thumbnail} 
                        alt="문제 썸네일" 
                        className="w-12 h-12 object-cover rounded"
                      />
                    )}
                    <div>
                      <p className="font-medium">{item.problemType}</p>
                      <p className="text-sm text-muted-foreground">
                        {item.timestamp.toLocaleDateString('ko-KR')} {item.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    <Badge variant={getScoreBadgeVariant(item.score)}>
                      {item.score}점
                    </Badge>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => onViewDetails(item.id)}
                    >
                      자세히 보기
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}