'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { toast } from 'sonner'
import { Play, Pause, RefreshCw, Activity, Zap } from 'lucide-react'
import { EEGChannelGrid } from './eeg-channel-grid'
import { EEGWaveform } from './eeg-waveform'
import { ElectrodeStatus } from './electrode-status'
import { PredictionDisplay } from './prediction-display'
import { TrainingInterface } from './training-interface'

interface EEGData {
  data: number[][]
  shape: [number, number]
  channels: number
  samples: number
  sampling_rate: number
  window_seconds: number
  electrode_status?: Record<string, any>
  data_range: {
    min: number
    max: number
    mean: number
    std: number
  }
}

interface PredictionResult {
  probs: number[]
  n_chans: number
  n_times: number
  window_seconds: number
  electrode_status: Record<string, any>
  synthetic_data?: boolean
  data_source?: string
}

export function EEGDashboard() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [eegData, setEegData] = useState<EEGData | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [dataSource, setDataSource] = useState<'synthetic' | 'real'>('synthetic')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  // Fetch synthetic data
  const fetchSyntheticData = useCallback(async () => {
    try {
      setIsLoading(true)
      console.log('Fetching data from:', `${apiUrl}/synthetic-data`)
      
      const response = await fetch(`${apiUrl}/synthetic-data`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data: EEGData = await response.json()
      console.log('Received EEG data:', data.shape)
      setEegData(data)
      
      // Also get prediction
      try {
        const predResponse = await fetch(`${apiUrl}/predict-synthetic`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          mode: 'cors',
        })
        
        if (predResponse.ok) {
          const predData: PredictionResult = await predResponse.json()
          console.log('Received prediction:', predData.probs)
          setPrediction(predData)
        }
      } catch (predError) {
        console.warn('Prediction failed:', predError)
        // Don't fail the whole process if prediction fails
      }
      
      toast.success('Synthetic EEG data loaded successfully')
    } catch (error) {
      console.error('Error fetching synthetic data:', error)
      toast.error(`Failed to fetch data: ${error instanceof Error ? error.message : 'Unknown error'}. Check if API is running on ${apiUrl}`)
    } finally {
      setIsLoading(false)
    }
  }, [apiUrl])

  // Fetch streaming data
  const fetchStreamData = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/synthetic-stream`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data: EEGData = await response.json()
      setEegData(data)
    } catch (error) {
      console.error('Error fetching stream data:', error)
      toast.error(`Stream error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }, [apiUrl])

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh && isStreaming) {
      const interval = setInterval(() => {
        if (dataSource === 'synthetic') {
          fetchStreamData()
        }
      }, 1000) // Update every second

      return () => clearInterval(interval)
    }
  }, [autoRefresh, isStreaming, dataSource, fetchStreamData])

  // Start/stop streaming
  const toggleStreaming = () => {
    if (isStreaming) {
      setIsStreaming(false)
      toast.info('Streaming stopped')
    } else {
      setIsStreaming(true)
      toast.success('Streaming started')
      if (dataSource === 'synthetic') {
        fetchSyntheticData()
      }
    }
  }

  // Manual refresh
  const handleRefresh = () => {
    if (dataSource === 'synthetic') {
      fetchSyntheticData()
    }
  }

  // Check API health
  const checkApiHealth = async () => {
    try {
      console.log('Checking API health at:', `${apiUrl}/health`)
      const response = await fetch(`${apiUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      })
      
      if (response.ok) {
        const health = await response.json()
        console.log('API health response:', health)
        toast.success(`API is healthy! Weights loaded: ${health.weights_loaded}`)
      } else {
        toast.error(`API health check failed: ${response.status} ${response.statusText}`)
      }
    } catch (error) {
      console.error('API health check error:', error)
      toast.error(`Cannot connect to API at ${apiUrl}. Is the server running?`)
    }
  }

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            EEG Control Panel
          </CardTitle>
          <CardDescription>
            Configure data source and streaming settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* API URL */}
            <div className="space-y-2">
              <label className="text-sm font-medium">API URL</label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm"
                placeholder="http://localhost:8000"
              />
            </div>

            {/* Data Source */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Data Source</label>
              <Select value={dataSource} onValueChange={(value: 'synthetic' | 'real') => setDataSource(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="synthetic">Synthetic Board</SelectItem>
                  <SelectItem value="real">Real Cyton Board</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Auto Refresh */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Auto Refresh</label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={autoRefresh}
                  onCheckedChange={setAutoRefresh}
                />
                <span className="text-sm text-slate-600">
                  {autoRefresh ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>

            {/* Status */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Status</label>
              <Badge variant={isStreaming ? 'default' : 'secondary'} className="w-fit">
                {isStreaming ? 'Streaming' : 'Stopped'}
              </Badge>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2">
            <Button onClick={toggleStreaming} disabled={isLoading}>
              {isStreaming ? (
                <>
                  <Pause className="h-4 w-4 mr-2" />
                  Stop Stream
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Stream
                </>
              )}
            </Button>

            <Button variant="outline" onClick={handleRefresh} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>

            <Button variant="outline" onClick={checkApiHealth}>
              <Zap className="h-4 w-4 mr-2" />
              Check API
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Data Display */}
      {eegData && (
        <div className="grid grid-cols-1 gap-6">
          {/* Channel Grid */}
          <Card>
            <CardHeader>
              <CardTitle>Channel Overview</CardTitle>
              <CardDescription>
                {eegData.channels} channels × {eegData.samples} samples ({eegData.window_seconds}s)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <EEGChannelGrid data={eegData.data} />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Electrode Status and Predictions */}
      {(eegData?.electrode_status || prediction) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Electrode Status */}
          {eegData?.electrode_status && (
            <Card>
              <CardHeader>
                <CardTitle>Electrode Status</CardTitle>
                <CardDescription>
                  Connection quality and impedance monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ElectrodeStatus status={eegData.electrode_status} />
              </CardContent>
            </Card>
          )}

          {/* Predictions */}
          {prediction && (
            <Card>
              <CardHeader>
                <CardTitle>Meditation Detection</CardTitle>
                <CardDescription>
                  Real-time meditation state classification
                  {prediction.synthetic_data && (
                    <Badge variant="outline" className="ml-2">
                      Synthetic Data
                    </Badge>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <PredictionDisplay prediction={prediction} />
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Training Interface */}
      <TrainingInterface />

      {/* Data Statistics */}
      {eegData && (
        <Card>
          <CardHeader>
            <CardTitle>Data Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {eegData.data_range.min.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Min Value</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {eegData.data_range.max.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Max Value</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {eegData.data_range.mean.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Mean</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {eegData.data_range.std.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Std Dev</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
