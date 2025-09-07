'use client'

import { useState, useEffect } from 'react'
import { EEGDashboard } from '@/components/eeg-dashboard'
import { Toaster } from '@/components/ui/sonner'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            EEG Monitor
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            Real-time brain signal visualization with LaBraM classification
          </p>
        </div>
        
        <EEGDashboard />
      </div>
      
      <Toaster />
    </main>
  )
}