'use client'

interface EEGChannelGridProps {
  data: number[][]
}

export function EEGChannelGrid({ data }: EEGChannelGridProps) {
  const channelNames = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
  
  const getChannelColor = (channelData: number[]) => {
    const std = Math.sqrt(channelData.reduce((sum, val) => sum + val * val, 0) / channelData.length)
    
    if (std < 10) return 'bg-red-100 border-red-300 text-red-800'
    if (std > 100) return 'bg-yellow-100 border-yellow-300 text-yellow-800'
    return 'bg-green-100 border-green-300 text-green-800'
  }

  const getActivityLevel = (channelData: number[]) => {
    const std = Math.sqrt(channelData.reduce((sum, val) => sum + val * val, 0) / channelData.length)
    return Math.min(100, (std / 50) * 100)
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {data.slice(0, 8).map((channelData, index) => {
        const activity = getActivityLevel(channelData)
        const colorClass = getChannelColor(channelData)
        
        return (
          <div
            key={index}
            className={`p-3 rounded-lg border-2 transition-all duration-300 ${colorClass}`}
          >
            <div className="text-center">
              <div className="font-semibold text-sm mb-1">
                {channelNames[index]}
              </div>
              <div className="text-xs opacity-75 mb-2">
                Ch {index + 1}
              </div>
              
              {/* Activity Bar */}
              <div className="w-full bg-white bg-opacity-50 rounded-full h-2 mb-2">
                <div
                  className="bg-current h-2 rounded-full transition-all duration-500"
                  style={{ width: `${activity}%` }}
                />
              </div>
              
              {/* Signal Stats */}
              <div className="text-xs space-y-1">
                <div>
                  Range: {Math.min(...channelData).toFixed(1)} to {Math.max(...channelData).toFixed(1)}
                </div>
                <div>
                  RMS: {Math.sqrt(channelData.reduce((sum, val) => sum + val * val, 0) / channelData.length).toFixed(1)}
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
