'use client'

import { Badge } from '@/components/ui/badge'
import { Brain, TrendingUp } from 'lucide-react'

interface PredictionDisplayProps {
  prediction: {
    probs: number[]
    n_chans: number
    n_times: number
    window_seconds: number
    electrode_status: Record<string, any>
    synthetic_data?: boolean
    data_source?: string
  }
}

export function PredictionDisplay({ prediction }: PredictionDisplayProps) {
  const { probs } = prediction
  const maxProb = Math.max(...probs)
  const predictedClass = probs.indexOf(maxProb)
  
  // Class labels - you can customize these based on your model
  const classLabels = probs.map((_, i) => `Class ${i + 1}`)
  
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50'
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const getBarColor = (prob: number, isMax: boolean) => {
    if (isMax) {
      if (prob >= 0.8) return 'bg-green-500'
      if (prob >= 0.6) return 'bg-yellow-500'
      return 'bg-red-500'
    }
    return 'bg-gray-300'
  }

  return (
    <div className="space-y-6">
      {/* Main Prediction */}
      <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Brain className="h-6 w-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-blue-900">
            Predicted Class
          </h3>
        </div>
        
        <div className="text-3xl font-bold text-blue-600 mb-2">
          {classLabels[predictedClass]}
        </div>
        
        <div className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(maxProb)}`}>
          <TrendingUp className="h-4 w-4" />
          Confidence: {(maxProb * 100).toFixed(1)}%
        </div>
      </div>

      {/* Probability Breakdown */}
      <div className="space-y-3">
        <h4 className="font-semibold text-gray-900">Class Probabilities</h4>
        
        {probs.map((prob, index) => (
          <div key={index} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">
                {classLabels[index]}
              </span>
              <span className="text-sm text-gray-600">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-500 ${getBarColor(prob, index === predictedClass)}`}
                style={{ width: `${prob * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Prediction Metadata */}
      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900">
            {prediction.n_chans}
          </div>
          <div className="text-sm text-gray-600">Channels</div>
        </div>
        
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900">
            {prediction.window_seconds}s
          </div>
          <div className="text-sm text-gray-600">Window</div>
        </div>
      </div>

      {/* Data Source Info */}
      {prediction.synthetic_data && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Badge variant="outline" className="text-xs">
            Synthetic Data
          </Badge>
          {prediction.data_source && (
            <Badge variant="secondary" className="text-xs">
              {prediction.data_source}
            </Badge>
          )}
        </div>
      )}
    </div>
  )
}
