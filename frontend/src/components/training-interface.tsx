"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { toast } from 'sonner'
import { Play, Square, Settings, TrendingUp, Clock, Zap } from 'lucide-react'

interface TrainingConfig {
  n_epochs: number
  n_samples_per_class: number
  val_split: number
  freeze_backbone: boolean
  learning_rate: number
  batch_size: number
  weight_decay: number
  save_dir: string
}

interface TrainingStatus {
  is_training: boolean
  current_epoch: number
  total_epochs: number
  train_loss: number
  val_loss: number
  train_acc: number
  val_acc: number
  best_val_acc: number
  start_time: number | null
  elapsed_time: number
  elapsed_time_formatted?: string
  eta?: number
  eta_formatted?: string
  config: TrainingConfig | null
  history: {
    train_losses: number[]
    val_losses: number[]
    train_accs: number[]
    val_accs: number[]
  }
  error?: string
}

export function TrainingInterface() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [config, setConfig] = useState<TrainingConfig>({
    n_epochs: 15,
    n_samples_per_class: 1500,
    val_split: 0.2,
    freeze_backbone: true,
    learning_rate: 0.001,
    batch_size: 32,
    weight_decay: 0.0001,
    save_dir: "./trained_models"
  })
  const [isConfigOpen, setIsConfigOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  // Poll training status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/training/status')
        if (response.ok) {
          const data = await response.json()
          setStatus(data)
        }
      } catch (error) {
        console.error('Failed to fetch training status:', error)
      }
    }

    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000)
    pollStatus() // Initial fetch

    return () => clearInterval(interval)
  }, [])

  const startTraining = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      })

      if (response.ok) {
        const result = await response.json()
        toast.success('Training started successfully!')
        setIsConfigOpen(false)
      } else {
        const error = await response.json()
        toast.error(`Failed to start training: ${error.detail}`)
      }
    } catch (error) {
      toast.error('Failed to start training')
      console.error('Training start error:', error)
    } finally {
      setLoading(false)
    }
  }

  const stopTraining = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/training/stop', {
        method: 'POST'
      })

      if (response.ok) {
        toast.success('Training stop requested')
      } else {
        const error = await response.json()
        toast.error(`Failed to stop training: ${error.detail}`)
      }
    } catch (error) {
      toast.error('Failed to stop training')
      console.error('Training stop error:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatPercentage = (value: number) => `${value.toFixed(1)}%`
  const formatLoss = (value: number) => value.toFixed(4)

  return (
    <div className="space-y-6">
      {/* Training Control Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Model Training
              </CardTitle>
              <CardDescription>
                Train meditation detection model on synthetic EEG data
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Dialog open={isConfigOpen} onOpenChange={setIsConfigOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Settings className="h-4 w-4 mr-2" />
                    Config
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>Training Configuration</DialogTitle>
                    <DialogDescription>
                      Adjust training parameters for the meditation detection model
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Epochs</label>
                      <input
                        type="number"
                        value={config.n_epochs}
                        onChange={(e) => setConfig({...config, n_epochs: parseInt(e.target.value)})}
                        className="w-full px-3 py-2 border rounded-md"
                        min="1"
                        max="100"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Samples per Class</label>
                      <input
                        type="number"
                        value={config.n_samples_per_class}
                        onChange={(e) => setConfig({...config, n_samples_per_class: parseInt(e.target.value)})}
                        className="w-full px-3 py-2 border rounded-md"
                        min="100"
                        max="5000"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Validation Split</label>
                      <input
                        type="number"
                        step="0.05"
                        value={config.val_split}
                        onChange={(e) => setConfig({...config, val_split: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 border rounded-md"
                        min="0.1"
                        max="0.5"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Batch Size</label>
                      <Select value={config.batch_size.toString()} onValueChange={(value) => setConfig({...config, batch_size: parseInt(value)})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="16">16</SelectItem>
                          <SelectItem value="32">32</SelectItem>
                          <SelectItem value="64">64</SelectItem>
                          <SelectItem value="128">128</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Learning Rate</label>
                      <Select value={config.learning_rate.toString()} onValueChange={(value) => setConfig({...config, learning_rate: parseFloat(value)})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.01">0.01</SelectItem>
                          <SelectItem value="0.005">0.005</SelectItem>
                          <SelectItem value="0.001">0.001</SelectItem>
                          <SelectItem value="0.0005">0.0005</SelectItem>
                          <SelectItem value="0.0001">0.0001</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="freeze-backbone"
                        checked={config.freeze_backbone}
                        onCheckedChange={(checked) => setConfig({...config, freeze_backbone: checked})}
                      />
                      <label htmlFor="freeze-backbone" className="text-sm font-medium">
                        Freeze Backbone (faster training)
                      </label>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2 pt-4">
                    <Button variant="outline" onClick={() => setIsConfigOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={startTraining} disabled={loading || status?.is_training}>
                      <Play className="h-4 w-4 mr-2" />
                      Start Training
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
              
              {status?.is_training ? (
                <Button onClick={stopTraining} disabled={loading} variant="destructive" size="sm">
                  <Square className="h-4 w-4 mr-2" />
                  Stop
                </Button>
              ) : (
                <Button onClick={() => setIsConfigOpen(true)} disabled={loading} size="sm">
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Status Badge */}
            <div className="flex items-center gap-2">
              <Badge variant={status?.is_training ? "default" : "secondary"}>
                {status?.is_training ? "Training" : "Idle"}
              </Badge>
              {status?.error && (
                <Badge variant="destructive">Error: {status.error}</Badge>
              )}
            </div>

            {/* Training Progress */}
            {status?.is_training && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Epoch {status.current_epoch} of {status.total_epochs}</span>
                    <span>{Math.round((status.current_epoch / status.total_epochs) * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(status.current_epoch / status.total_epochs) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Training Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{formatLoss(status.train_loss)}</div>
                    <div className="text-xs text-gray-600">Train Loss</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{formatPercentage(status.train_acc)}</div>
                    <div className="text-xs text-gray-600">Train Acc</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{formatLoss(status.val_loss)}</div>
                    <div className="text-xs text-gray-600">Val Loss</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">{formatPercentage(status.val_acc)}</div>
                    <div className="text-xs text-gray-600">Val Acc</div>
                  </div>
                </div>

                {/* Time Information */}
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    Elapsed: {status.elapsed_time_formatted || '0s'}
                  </div>
                  {status.eta_formatted && (
                    <div className="flex items-center gap-1">
                      <Zap className="h-4 w-4" />
                      ETA: {status.eta_formatted}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Best Model Info */}
            {status?.best_val_acc > 0 && (
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="text-sm font-medium text-green-800">
                  Best Model: {formatPercentage(status.best_val_acc)} validation accuracy
                </div>
                <div className="text-xs text-green-600 mt-1">
                  Saved to {status.config?.save_dir || './trained_models'}
                </div>
              </div>
            )}

            {/* Training Configuration Display */}
            {status?.config && (
              <div className="text-xs text-gray-600 space-y-1">
                <div className="font-medium">Current Configuration:</div>
                <div className="grid grid-cols-2 gap-2">
                  <div>Epochs: {status.config.n_epochs}</div>
                  <div>Samples: {status.config.n_samples_per_class}/class</div>
                  <div>Batch Size: {status.config.batch_size}</div>
                  <div>Learning Rate: {status.config.learning_rate}</div>
                  <div>Freeze Backbone: {status.config.freeze_backbone ? 'Yes' : 'No'}</div>
                  <div>Val Split: {(status.config.val_split * 100).toFixed(0)}%</div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
