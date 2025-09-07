'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'

interface EEGWaveformProps {
  data: number[][]
  samplingRate: number
  channels: number
}

export function EEGWaveform({ data, samplingRate, channels }: EEGWaveformProps) {
  const channelNames = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
  const colors = [
    '#ef4444', // red
    '#f97316', // orange  
    '#eab308', // yellow
    '#22c55e', // green
    '#06b6d4', // cyan
    '#3b82f6', // blue
    '#8b5cf6', // violet
    '#ec4899', // pink
  ]

  // Convert data to chart format - sample every 10th point for performance
  const chartData = data[0]?.map((_, timeIndex) => {
    if (timeIndex % 10 !== 0) return null // Sample every 10th point
    
    const timeSeconds = timeIndex / samplingRate
    const point: any = { time: timeSeconds.toFixed(3) }
    
    for (let ch = 0; ch < Math.min(channels, data.length); ch++) {
      point[channelNames[ch]] = data[ch][timeIndex]
    }
    
    return point
  }).filter(Boolean) || []

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
          <XAxis 
            dataKey="time" 
            type="number"
            scale="linear"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(value) => `${value}s`}
          />
          <YAxis 
            tickFormatter={(value) => `${value.toFixed(1)}`}
          />
          <Legend />
          
          {channelNames.slice(0, Math.min(channels, data.length)).map((name, index) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={colors[index]}
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
