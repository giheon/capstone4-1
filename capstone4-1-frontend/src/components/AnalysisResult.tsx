import React from 'react';
import { CheckCircle, XCircle, AlertCircle, BookOpen } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';

interface AnalysisStep {
  step: number;
  description: string;
  userAnswer: string;
  isCorrect: boolean;
  correctAnswer?: string;
  explanation?: string;
}

interface AnalysisResultProps {
  ocrText: string;
  detectedProblem: string;
  userSolution: string;
  analysisSteps: AnalysisStep[];
  overallScore: number;
  isLoading?: boolean;
}

export function AnalysisResult({ 
  ocrText, 
  detectedProblem, 
  userSolution, 
  analysisSteps, 
  overallScore,
  isLoading = false 
}: AnalysisResultProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              분석 중...
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              업로드된 이미지를 분석하고 있습니다. 잠시만 기다려주세요.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="space-y-6">
      {/* OCR 결과 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            문제 인식 결과
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="mb-2">감지된 문제:</h4>
            <div className="bg-muted p-3 rounded-lg">
              <p className="font-mono">{detectedProblem}</p>
            </div>
          </div>
          
          <div>
            <h4 className="mb-2">사용자 풀이:</h4>
            <div className="bg-muted p-3 rounded-lg">
              <p className="font-mono whitespace-pre-wrap">{userSolution}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 전체 점수 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>분석 결과</span>
            <Badge variant={getScoreBadgeVariant(overallScore)} className="text-lg px-3 py-1">
              {overallScore}점
            </Badge>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* 단계별 분석 */}
      <Card>
        <CardHeader>
          <CardTitle>단계별 분석</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {analysisSteps.map((step, index) => (
            <div key={step.step} className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-1">
                  {step.isCorrect ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-600" />
                  )}
                </div>
                
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">단계 {step.step}</span>
                    <Badge variant={step.isCorrect ? 'default' : 'destructive'} className="text-xs">
                      {step.isCorrect ? '정답' : '오답'}
                    </Badge>
                  </div>
                  
                  <p className="text-muted-foreground">{step.description}</p>
                  
                  <div className="bg-muted p-3 rounded-lg">
                    <p className="font-mono text-sm">{step.userAnswer}</p>
                  </div>
                  
                  {!step.isCorrect && step.correctAnswer && (
                    <div className="bg-green-50 border border-green-200 p-3 rounded-lg dark:bg-green-950 dark:border-green-800">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertCircle className="h-4 w-4 text-green-600" />
                        <span className="text-sm font-medium text-green-800 dark:text-green-200">올바른 답:</span>
                      </div>
                      <p className="font-mono text-sm text-green-700 dark:text-green-300">{step.correctAnswer}</p>
                      {step.explanation && (
                        <p className="text-sm text-green-600 dark:text-green-400 mt-2">{step.explanation}</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              {index < analysisSteps.length - 1 && <Separator />}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}