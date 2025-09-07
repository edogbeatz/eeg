'use client'

import { Badge } from '@/components/ui/badge'
import { CheckCircle, AlertTriangle, XCircle, Wifi } from 'lucide-react'

interface ElectrodeStatusProps {
  status: Record<string, {
    status: string
    quality: number
    std: number
    range: number
    energy: number
  }>
}

export function ElectrodeStatus({ status }: ElectrodeStatusProps) {
  const getStatusIcon = (statusStr: string, quality: number) => {
    switch (statusStr) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'poor_contact':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />
      case 'noisy':
        return <Wifi className="h-4 w-4 text-orange-600" />
      case 'disconnected':
      default:
        return <XCircle className="h-4 w-4 text-red-600" />
    }
  }

  const getStatusColor = (statusStr: string) => {
    switch (statusStr) {
      case 'connected':
        return 'bg-green-50 border-green-200'
      case 'poor_contact':
        return 'bg-yellow-50 border-yellow-200'
      case 'noisy':
        return 'bg-orange-50 border-orange-200'
      case 'disconnected':
      default:
        return 'bg-red-50 border-red-200'
    }
  }

  const getBadgeVariant = (statusStr: string) => {
    switch (statusStr) {
      case 'connected':
        return 'default'
      case 'poor_contact':
        return 'secondary'
      case 'noisy':
        return 'outline'
      case 'disconnected':
      default:
        return 'destructive'
    }
  }

  const channelNames = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.values(status).reduce((acc, curr) => {
          const existing = acc.find(item => item.status === curr.status)
          if (existing) {
            existing.count++
          } else {
            acc.push({ status: curr.status, count: 1 })
          }
          return acc
        }, [] as Array<{ status: string; count: number }>).map(({ status: statusStr, count }) => (
          <Badge key={statusStr} variant={getBadgeVariant(statusStr)}>
            {statusStr.replace('_', ' ')}: {count}
          </Badge>
        ))}
      </div>

      {/* Individual Channels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Object.entries(status).map(([channel, data], index) => (
          <div
            key={channel}
            className={`p-4 rounded-lg border ${getStatusColor(data.status)} transition-all duration-300`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getStatusIcon(data.status, data.quality)}
                <span className="font-semibold">
                  {channelNames[index] || channel}
                </span>
              </div>
              <Badge variant={getBadgeVariant(data.status)} className="text-xs">
                {data.status.replace('_', ' ')}
              </Badge>
            </div>

            {/* Quality Bar */}
            <div className="mb-3">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>Quality</span>
                <span>{(data.quality * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    data.quality >= 0.8 ? 'bg-green-500' :
                    data.quality >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${data.quality * 100}%` }}
                />
              </div>
            </div>

            {/* Signal Metrics */}
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="text-center">
                <div className="font-medium">{data.std.toFixed(1)}</div>
                <div className="text-gray-500">Std Dev</div>
              </div>
              <div className="text-center">
                <div className="font-medium">{data.range.toFixed(1)}</div>
                <div className="text-gray-500">Range</div>
              </div>
              <div className="text-center">
                <div className="font-medium">{data.energy.toFixed(1)}</div>
                <div className="text-gray-500">Energy</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
